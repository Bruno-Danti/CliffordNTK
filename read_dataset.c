#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void read_sparse_dataset(
    const char *path,
    float ***dataset,
    int n_images,
    int vector_length
){
    FILE *f = fopen(path, "rb");
    if (!f) {perror("open"); exit(1);}

    *dataset = malloc(n_images * sizeof(float*));
    for (int i = 0; i < n_images; i++)
    {
        (*dataset)[i] = calloc(vector_length, sizeof(float));
        
        uint32_t n;
        fread(&n, sizeof(uint32_t), 1, f);
        
        uint16_t *indices = malloc(n * sizeof(uint16_t));
        float *values = malloc(n * sizeof(float));
        
        fread(indices, sizeof(uint16_t), n, f);
        fread(values, sizeof(float), n, f);

        for (uint32_t j = 0; j < n; j++)
        {
            (*dataset)[i][indices[j]] = values[j];
        }
        
        free(indices);
        free(values);
    }
    fclose(f);
}


int main(void){
    int n_images = 9;
    int vec_length = 1024;
    float **dataset;
    read_sparse_dataset("encoded_dataset", &dataset, n_images, vec_length);

    for (int i = 0; i < n_images; i++)
    {
        int n_nonzero_elements = 0;
        for (int j = 8; j < vec_length; j++)
        {
            if (dataset[i][j] != 0.0)
            {
                n_nonzero_elements += 1;
                printf("%i: %f\n", j, dataset[i][j]);
            }
            
        }
        printf("%i\n", n_nonzero_elements);
    }
    
    



    printf("Success\n");
}