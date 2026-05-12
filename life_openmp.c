#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Required for OpenMP

// Function to print the grid
void print_grid(int *grid, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", grid[i * cols + j]);
        }
        printf("\n");
    }
    printf("-------------------\n");
}

// Function to count alive neighbors around a specific cell
int count_neighbors(int *grid, int r, int c, int rows, int cols) {
    int count = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (i == 0 && j == 0) continue; // Skip the cell itself
            
            int nr = r + i;
            int nc = c + j;
            
            // Check if neighbor is within boundaries
            if (nr >= 0 && nr < rows && nc >= 0 && nc < cols) {
                count += grid[nr * cols + nc];
            }
        }
    }
    return count;
}

int main() {
    int rows = 10, cols = 10;
    int generations = 3;

    // 1D array initialization initialized to 0 using calloc
    int *grid = (int*)calloc(rows * cols, sizeof(int));
    int *next_grid = (int*)calloc(rows * cols, sizeof(int));

    /* --- CSV FILE READING LOGIC (Uncomment for the exam) ---
    FILE *file = fopen("input.csv", "r");
    if (file != NULL) {
        char line[256];
        // Assuming CSV format is: row,col
        while (fgets(line, sizeof(line), file)) {
            int r, c;
            sscanf(line, "%d,%d", &r, &c);
            grid[r * cols + c] = 1; // Set cell to alive
        }
        fclose(file);
    } 
    ------------------------------------------------------- */

    // For testing right now, let's manually place a "Glider" pattern
    grid[0 * cols + 1] = 1;
    grid[1 * cols + 2] = 1;
    grid[2 * cols + 0] = 1;
    grid[2 * cols + 1] = 1;
    grid[2 * cols + 2] = 1;

    printf("Initial Grid:\n");
    print_grid(grid, rows, cols);

    // Simulation Loop
    for (int gen = 0; gen < generations; gen++) {
        
        // OPENMP MAGIC: This single line divides the rows among your CPU threads
        #pragma omp parallel for
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int neighbors = count_neighbors(grid, i, j, rows, cols);

                // Game of Life Rules
                if (grid[i * cols + j] == 1 && (neighbors == 2 || neighbors == 3)) {
                    next_grid[i * cols + j] = 1; // Survives
                } else if (grid[i * cols + j] == 0 && neighbors == 3) {
                    next_grid[i * cols + j] = 1; // Born
                } else {
                    next_grid[i * cols + j] = 0; // Dies
                }
            }
        }

        // Copy next_grid back to grid for the next generation
        for (int i = 0; i < rows * cols; i++) {
            grid[i] = next_grid[i];
        }
    }

    printf("Final Grid after %d generations:\n", generations);
    print_grid(grid, rows, cols);

    // Clean up memory
    free(grid);
    free(next_grid);
    return 0;
}