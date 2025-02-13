-- ================================================================
-- Full Query: Absorbing Markov Chain Attribution with Fundamental Matrix
-- ================================================================

-- UDF: Invert a square matrix using Gauss-Jordan elimination.
CREATE TEMP FUNCTION invertMatrix(matrix ANY TYPE)
RETURNS ARRAY<ARRAY<FLOAT64>>
LANGUAGE js AS """
function invertMatrix(matrix) {
  const n = matrix.length;
  // Create copies of matrix (A) and identity matrix (I)
  let A = matrix.map(row => row.slice());
  let I = [];
  for (let i = 0; i < n; i++) {
    I[i] = [];
    for (let j = 0; j < n; j++) {
      I[i][j] = (i === j ? 1.0 : 0.0);
    }
  }
  // Perform Gauss-Jordan elimination.
  for (let i = 0; i < n; i++) {
    let pivot = A[i][i];
    if (Math.abs(pivot) < 1e-12) {
      throw new Error("Matrix is singular");
    }
    for (let j = 0; j < n; j++) {
      A[i][j] /= pivot;
      I[i][j] /= pivot;
    }
    for (let k = 0; k < n; k++) {
      if (k !== i) {
        let factor = A[k][i];
        for (let j = 0; j < n; j++) {
          A[k][j] -= factor * A[i][j];
          I[k][j] -= factor * I[i][j];
        }
      }
    }
  }
  return I;
}
return invertMatrix(matrix);
""";

-- UDF: Multiply two matrices.
CREATE TEMP FUNCTION multiplyMatrices(matrixA ANY TYPE, matrixB ANY TYPE)
RETURNS ARRAY<ARRAY<FLOAT64>>
LANGUAGE js AS """
function multiplyMatrices(A, B) {
  const n = A.length;
  const m = B[0].length;
  const p = A[0].length;
  let result = [];
  for (let i = 0; i < n; i++) {
    result[i] = [];
    for (let j = 0; j < m; j++) {
      let sum = 0;
      for (let k = 0; k < p; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}
return multiplyMatrices(matrixA, matrixB);
""";

-- ================================================================
-- Step 1. Build user journeys from event-level data.
-- (Assumes a table "pivoted" exists with:
--    blended_user_id, utm_source, session_start_ts, is_conversion_session, etc.)
-- ================================================================
WITH journeys AS (
  SELECT
    blended_user_id,
    ARRAY_CONCAT(['Start'], ARRAY_AGG(utm_source ORDER BY session_start_ts)) AS channel_sequence,
    MAX(CASE WHEN is_conversion_session THEN 1 ELSE 0 END) AS conversion
  FROM pivoted
  GROUP BY blended_user_id
),
journeys_with_end AS (
  SELECT
    blended_user_id,
    IF(conversion = 1,
       ARRAY_CONCAT(channel_sequence, ['Conversion']),
       ARRAY_CONCAT(channel_sequence, ['Null'])
    ) AS channel_sequence
  FROM journeys
),
-- ================================================================
-- Step 2. Expand journeys into state-to-state transitions.
-- ================================================================
transitions AS (
  SELECT
    blended_user_id,
    channel_sequence,
    channel_sequence[OFFSET(i)] AS from_state,
    channel_sequence[OFFSET(i + 1)] AS to_state
  FROM journeys_with_end,
       UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(channel_sequence) - 2)) AS i
),
-- ================================================================
-- Step 3. Count transitions and compute probabilities.
-- ================================================================
transition_counts AS (
  SELECT
    from_state,
    to_state,
    COUNT(*) AS cnt
  FROM transitions
  GROUP BY from_state, to_state
),
transition_probs AS (
  SELECT
    from_state,
    to_state,
    cnt,
    SUM(cnt) OVER (PARTITION BY from_state) AS total_from,
    SAFE_DIVIDE(cnt, SUM(cnt) OVER (PARTITION BY from_state)) AS prob
  FROM transition_counts
),
-- ================================================================
-- Step 4. Build the complete list of states.
-- ================================================================
state_list AS (
  SELECT state FROM (
    SELECT from_state AS state FROM transition_probs
    UNION DISTINCT
    SELECT to_state AS state FROM transition_probs
  )
),
ordered_states AS (
  SELECT ARRAY_AGG(state ORDER BY state) AS states FROM state_list
),
-- ================================================================
-- Step 5. Construct the full transition probability matrix as a 2D array.
-- Rows and columns follow the ordering in "ordered_states".
-- ================================================================
full_matrix AS (
  SELECT
    os.states,
    ARRAY(
      SELECT
        ARRAY(
          SELECT IFNULL(tp.prob, 0)
          FROM UNNEST(os.states) AS to_state
          LEFT JOIN transition_probs tp
            ON tp.from_state = from_state AND tp.to_state = to_state
          ORDER BY to_state
        )
      FROM UNNEST(os.states) AS from_state
      ORDER BY from_state
    ) AS matrix
  FROM ordered_states os
),
-- ================================================================
-- Step 6. Classify states as transient or absorbing.
-- We assume that only 'Conversion' and 'Null' are absorbing.
-- ================================================================
state_classification AS (
  SELECT
    state,
    CASE WHEN state IN ('Conversion', 'Null') THEN 'absorbing' ELSE 'transient' END AS type
  FROM state_list
),
transient_states AS (
  SELECT ARRAY_AGG(state ORDER BY state) AS states
  FROM state_classification
  WHERE type = 'transient'
),
absorbing_states AS (
  SELECT ARRAY_AGG(state ORDER BY state) AS states
  FROM state_classification
  WHERE type = 'absorbing'
),
-- ================================================================
-- Step 7. Bring together the full matrix and state arrays.
-- ================================================================
matrices AS (
  SELECT
    fm.states AS all_states,
    fm.matrix AS full_matrix,
    ts.states AS transient_states,
    abs.states AS absorbing_states
  FROM full_matrix fm
  CROSS JOIN transient_states ts
  CROSS JOIN absorbing_states abs
),
-- ================================================================
-- Step 8. Extract the Q and R matrices from the full transition matrix.
-- Q contains rows and columns for transient states;
-- R contains rows for transient states and columns for absorbing states.
-- Because arrays in BigQuery are 1D, we “simulate” submatrix extraction via array positions.
-- ================================================================
Q_R_extraction AS (
  SELECT
    m.all_states,
    m.full_matrix,
    m.transient_states,
    m.absorbing_states,
    -- Build Q: For each row index where the state is transient,
    -- take only the columns corresponding to transient states.
    ARRAY(
      SELECT
        (SELECT ARRAY_AGG(row_val ORDER BY col_idx)
         FROM UNNEST(m.full_matrix[OFFSET(idx)]) AS row_val WITH OFFSET col_idx
         WHERE m.all_states[OFFSET(col_idx)] IN UNNEST(m.transient_states)
        )
      FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(m.all_states)-1)) AS idx
      WHERE m.all_states[OFFSET(idx)] IN UNNEST(m.transient_states)
    ) AS Q,
    -- Build R: For each transient state row, take the columns corresponding to absorbing states.
    ARRAY(
      SELECT
        (SELECT ARRAY_AGG(row_val ORDER BY col_idx)
         FROM UNNEST(m.full_matrix[OFFSET(idx)]) AS row_val WITH OFFSET col_idx
         WHERE m.all_states[OFFSET(col_idx)] IN UNNEST(m.absorbing_states)
        )
      FROM UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(m.all_states)-1)) AS idx
      WHERE m.all_states[OFFSET(idx)] IN UNNEST(m.transient_states)
    ) AS R
  FROM matrices m
),
-- ================================================================
-- Step 9. Compute I - Q.
-- First, build an identity matrix I of the same size as Q.
-- ================================================================
I_minus_Q AS (
  SELECT
    (SELECT ARRAY(
       SELECT ARRAY(
         SELECT IF(i = j, 1.0, 0.0)
         FROM UNNEST(GENERATE_ARRAY(0, q_size-1)) AS j
       )
       FROM UNNEST(GENERATE_ARRAY(0, q_size-1)) AS i
     )) AS I_matrix,
    Q
  FROM (
    SELECT ARRAY_LENGTH(Q) AS q_size, Q
    FROM Q_R_extraction
    LIMIT 1
  )
),
I_subtract_Q AS (
  SELECT
    ARRAY(
      SELECT
        ARRAY(
          SELECT I_matrix[OFFSET(i)][OFFSET(j)] - Q[OFFSET(i)][OFFSET(j)]
          FROM UNNEST(GENERATE_ARRAY(0, q_size-1)) AS j
        )
      FROM UNNEST(GENERATE_ARRAY(0, q_size-1)) AS i
    ) AS I_minus_Q_matrix
  FROM (
    SELECT q_size, I_matrix, Q
    FROM I_minus_Q
  )
),
-- ================================================================
-- Step 10. Compute the fundamental matrix N = (I - Q)^(-1)
-- ================================================================
N_matrix AS (
  SELECT invertMatrix(I_minus_Q_matrix) AS N
  FROM I_subtract_Q
),
-- ================================================================
-- Step 11. Compute absorption probabilities: B = N * R.
-- ================================================================
B_matrix AS (
  SELECT multiplyMatrices(N_matrix.N, extraction.R) AS B
  FROM N_matrix,
       (SELECT R FROM Q_R_extraction LIMIT 1) extraction
),
-- ================================================================
-- Step 12. Flatten the B matrix to show, for each transient state,
-- its probability of being absorbed in each absorbing state.
-- ================================================================
absorption_results AS (
  SELECT
    transient_state,
    absorbing_state,
    B_matrix.B[transient_idx][absorbing_idx] AS absorption_probability
  FROM B_matrix,
       (SELECT transient_states, absorbing_states FROM Q_R_extraction LIMIT 1),
       UNNEST(transient_states) AS transient_state WITH OFFSET transient_idx,
       UNNEST(absorbing_states) AS absorbing_state WITH OFFSET absorbing_idx
)
-- ================================================================
-- Final Output: Absorption probabilities for each transient state.
-- (These represent the estimated “attribution” probabilities.)
-- ================================================================
SELECT
  transient_state,
  absorbing_state,
  absorption_probability
FROM absorption_results
ORDER BY transient_state, absorbing_state;
