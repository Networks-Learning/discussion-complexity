# On complexity of online discussions

This is the code accompanying the paper:

> On complexity of online discussions. Utkarsh Upadhyay, Abir De, Aasish Pappu, Manuel Gomez-Rodriguez. WSDM, 2018.

The code is arranged as various scripts which do a variety of tasks which were used in the paper.
The key scripts are:

  1. Determining whether a discussion has a given dimension: [`2d_sat.py`](#2d_sat.py)
  2. Filling in the comment-voter matrix using:
    1. our proposed polynomial algorithm: [`SR_fill.py`](#SR_fill.py)
    2. Z3 embeddings: [`z3_fill.py`](#z3_fill.py)
    3. our proposed exact rank method: [`low-rank-completion.py`](#low-rank-completion.py)
    4. soft-impute: [`soft-impute.py`](#soft-impute.py)
  3. Finding the embeddings for a partial binary matrix using Z3: [`matrix-Z3-embed.py`](#matrix-Z3-embed.py)

## 2d_sat.py

    Usage: 2d_sat.py [OPTIONS] IN_FILE

      Reads data from IN_FILE with the following format:

            comment_tree_id, commenter_id, voter_id, vote_type
            1, 200, 3000, 1
            1, 201, 3000, -1
            ...

         or (in case of real data):

            r.reply_to        r.message_id    r.uid_alias     r.vote_type
            1         200     3000    UP
            1         201     3000    DOWN
            ...

      Outputs a CSV which contains whether the comment/article-tree had a sign-
      rank of 2 or not.

      Redirect output to a file to save it.

    Options:
      --dims INTEGER                  What dimensional embedding to test for?
      --cpus INTEGER                  How many CPUs to use.
      --timeout INTEGER               Time after which to give up (ms).
      --real / --no-real              Assume format of real-data.
      --improve PATH                  Improve the results from the provided file.
                                      Will only run for `unknown` ids in the file.
      --context-id / --no-context-id  Use the context_id instead of
                                      comment_tree_id to group comments into
                                      matrices.
      --nrows INTEGER                 Number of rows from the CSV to read.
      --help                          Show this message and exit.


## SR_fill.py

    Usage: SR_fill.py [OPTIONS] IN_MAT_FILE OP_MAT_FILE OP_SC_FILE

      Read M_partial from IN_MAT_FILE and fill in the matrix using SR. The vote
      at (i, j) will be removed before filling in the matrix. The resulting
      matrix will be saved at OP_MAT_FILE and the given LOO entry will be placed
      along with the original vote in OP_LOO_FILE.

      Additionally, the best guess for the Sign Rank will be placed in
      OP_SC_FILE along with the source node which results in that.

    Options:
      -i INTEGER                    LOO i
      -j INTEGER                    LOO j
      --op-loo PATH                 Output path for the LOO.
      --seed INTEGER                Seed which was used to create this test-case.
      --min-avg / --no-min-avg      This flag will cause minimization of the
                                    average SC instead of worst case SC. Is much
                                    faster.
      --transpose / --no-transpose  Whether to transpose the matrix or not.
      --help                        Show this message and exit.


## z3_fill.py

    Usage: z3_fill.py [OPTIONS] IN_MAT_FILE OP_MAT_FILE OP_LOO_FILE

      Read M_partial from IN_MAT_FILE and fill in the matrix using Z3. The vote
      at (i, j) will be removed before filling in the matrix. The resulting
      matrix will be saved at OP_MAT_FILE and the given LOO entry will be placed
      along with the original vote in OP_LOO_FILE.

    Options:
      -i INTEGER       LOO i
      -j INTEGER       LOO j
      --sat_2d TEXT    Is the matrix 2D-SAT w/o LOO?
      --sat_1d TEXT    Is the matrix 1D-SAT w/o LOO?
      --seed INTEGER   Seed which was used to create this test-cast.
      --guess INTEGER  Whether to use the given guess {-1, +1} for doing LOO
                       prediction; 0 means no guessing.
      --help           Show this message and exit.

## low-rank-completion.py

    Usage: low-rank-completion.py [OPTIONS] IN_MAT_FILE

      Read M_partial from IN_MAT_FILE and optimize the embeddings to maximize
      the likelihood under the logit model.

    Options:
      --dims INTEGER              The dimensionality of the embedding.
      --seed INTEGER              The random seed to use for initializing
                                  matrices, in case initial values are not given.
      --suffix TEXT               Suffix to add before saving the embeddings.
      --init-c-vecs TEXT          File which contains initial embedding of c_vecs.
      --init-v-vecs TEXT          File which contains initial embedding of v_vecs.
      -i INTEGER                  Which i index to LOO.
      -j INTEGER                  Which j index to LOO.
      --alpha FLOAT               Bound on the spikiness of M.
      --sigma FLOAT               What is the variance of (logistic) noise to add.
      --lbfgs / --no-lbfgs        Whether to use LBFGS instead of BFGS.
      --loo-output TEXT           Where to save the LOO output.
      --loo-only / --no-loo-only  Whether to only save the LOO output or whether
                                  to save the complete recovered matrix.
      --uv / --no-uv              Whether to impose the alpha constraint on both U
                                  and V or on U.V^T.
      --verbose / --no-verbose    Verbose output.
      --help                      Show this message and exit.

## soft-impute.py

    Usage: soft-impute.py [OPTIONS] IN_MAT_FILE

      Read M_partial from IN_MAT_FILE and complete the matrix using soft-impute
      method.

    Options:
      --dims INTEGER              The dimensionality of the embedding.
      --seed INTEGER              The random seed to use for initializing
                                  matrices, in case initial values are not given.
      --suffix TEXT               Suffix to add before saving the embeddings.
      -i INTEGER                  Which i index to LOO.
      -j INTEGER                  Which j index to LOO.
      --loo-output TEXT           Where to save the LOO output.
      --loo-only / --no-loo-only  Whether to only save the LOO output or whether
                                  to save the complete recovered matrix.
      --verbose / --no-verbose    Verbose output.
      --help                      Show this message and exit.

## matrix-Z3-embed.py

    Usage: matrix-Z3-embed.py [OPTIONS] MAT_FILE

      Read the partial matrix in MAT_FILE and save embeddings for the file to
      `mat_file.commenters` and `mat_file.voters` file.

    Options:
      --dim INTEGER      What dimension to use while splitting matrix.
      --timeout INTEGER  What timeout to use (minutes).
      --help             Show this message and exit.


# Requirements

These python packages are required:

  - `click`
  - `cvxpy`
  - `dccp `
  - `numpy`
  - `decorated_options`
  - `seaborn`
  - `z3-solver`
  - `networkx`
  - `pqdict`
  - `fancyimpute`

All of these can be installed using `pip` while some of them are available on
`conda` (preferred). If using `pip`, the `requirements.txt` file in the `code`
folder will be helpful.
