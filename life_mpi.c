#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Set your grid size here. Make sure ROWS is divisible by your number of processes!
#define ROWS 16 
#define COLS 16
#define GENERATIONS 5

// Function to count neighbors for a specific cell in the LOCAL grid
int count_neighbors(int *local_grid, int r, int c, int local_rows_with_ghosts, int cols) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue;
            
            int nr = r + i;
            int nc = c + j;
            
            if (nr >= 0 && nr < local_rows_with_ghosts && nc >= 0 && nc < cols) {
                count += local_grid[nr * cols + nc];
            }
        }
    }
    return count;
}

int main(int argc, char **argv) {
    int rank, size;

    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Safety check for clean division of rows
    if (ROWS % size != 0) {
        if (rank == 0) printf("Error: ROWS (%d) must be cleanly divisible by number of processes (%d).\n", ROWS, size);
        MPI_Finalize();
        return 1;
    }

    int local_rows = ROWS / size;
    
    // Allocate local grid with 2 extra rows for Top and Bottom Ghost Cells
    int *local_grid = (int*)calloc((local_rows + 2) * COLS, sizeof(int));
    int *local_next = (int*)calloc((local_rows + 2) * COLS, sizeof(int));

    int *global_grid = NULL;

    // ==========================================
    // RANK 0 ONLY: READ FROM CSV FILE
    // ==========================================
    if (rank == 0) {
        global_grid = (int*)calloc(ROWS * COLS, sizeof(int));
        
        FILE *infile = fopen("input.csv", "r");
        if (infile == NULL) {
            printf("Error: Could not open input.csv! Make sure the file exists.\n");
            MPI_Abort(MPI_COMM_WORLD, 1); // Kill all processes if file is missing
        }

        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                // Read integer, skip the comma
                fscanf(infile, "%d,", &global_grid[i * COLS + j]);
            }
        }
        fclose(infile);
        printf("Successfully read input.csv. Starting simulation...\n");
    }

    // 2. Scatter the data from Rank 0 to everyone
    // Start writing at &local_grid[COLS] to leave row 0 empty for the top ghost cell
    MPI_Scatter(global_grid, local_rows * COLS, MPI_INT, 
                &local_grid[COLS], local_rows * COLS, MPI_INT, 
                0, MPI_COMM_WORLD);

    // Identify neighbors
    int top_neighbor = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int bottom_neighbor = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    // 3. Main Simulation Loop
    for (int gen = 0; gen < GENERATIONS; gen++) {
        
        // --- GHOST CELL EXCHANGE ---
        // Send top real row to top neighbor, receive bottom ghost row
        MPI_Sendrecv(&local_grid[COLS], COLS, MPI_INT, top_neighbor, 0,
                     &local_grid[(local_rows + 1) * COLS], COLS, MPI_INT, bottom_neighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Send bottom real row to bottom neighbor, receive top ghost row
        MPI_Sendrecv(&local_grid[local_rows * COLS], COLS, MPI_INT, bottom_neighbor, 1,
                     &local_grid[0 * COLS], COLS, MPI_INT, top_neighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // --- COMPUTE GAME OF LIFE ---
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < COLS; j++) {
                int neighbors = count_neighbors(local_grid, i, j, local_rows + 2, COLS);

                if (local_grid[i * COLS + j] == 1 && (neighbors == 2 || neighbors == 3)) {
                    local_next[i * COLS + j] = 1; // Survive
                } else if (local_grid[i * COLS + j] == 0 && neighbors == 3) {
                    local_next[i * COLS + j] = 1; // Born
                } else {
                    local_next[i * COLS + j] = 0; // Die
                }
            }
        }

        // Update current grid
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 0; j < COLS; j++) {
                local_grid[i * COLS + j] = local_next[i * COLS + j];
            }
        }
    }

    // 4. Gather the computed rows back to Rank 0
    MPI_Gather(&local_grid[COLS], local_rows * COLS, MPI_INT, 
               global_grid, local_rows * COLS, MPI_INT, 
               0, MPI_COMM_WORLD);

    // ==========================================
    // RANK 0 ONLY: WRITE TO CSV FILE
    // ==========================================
    if (rank == 0) {
        FILE *outfile = fopen("output.csv", "w");
        if (outfile == NULL) {
            printf("Error creating output.csv!\n");
        } else {
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    fprintf(outfile, "%d", global_grid[i * COLS + j]);
                    if (j < COLS - 1) {
                        fprintf(outfile, ","); // Print comma between columns
                    }
                }
                fprintf(outfile, "\n"); // Newline at the end of the row
            }
            fclose(outfile);
            printf("Simulation complete. Results written to output.csv\n");
        }
        free(global_grid);
    }

    free(local_grid);
    free(local_next);
    
    MPI_Finalize();
    return 0;
}