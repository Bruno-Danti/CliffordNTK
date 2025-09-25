#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

void read_sparse_dataset(
    const char *path,
    float ***dataset,
    uint16_t ***indices,
    uint32_t **n_indices,
    int n_images,
    int vector_length
){
    FILE *f = fopen(path, "rb");
    if (!f) {perror("open"); exit(1);}

    *dataset = malloc(n_images * sizeof(float*));
    *indices = malloc(n_images * sizeof(uint16_t*));
    *n_indices = malloc(n_images * sizeof(uint32_t));

    for (int i = 0; i < n_images; i++)
    {
        (*dataset)[i] = calloc(vector_length, sizeof(float));
        
        // uint32_t n;
        // fread(&n, sizeof(uint32_t), 1, f);
        fread(&((*n_indices)[i]), sizeof(uint32_t), 1, f);
        uint32_t n = (*n_indices)[i];

        
        // uint16_t *indices = malloc(n * sizeof(uint16_t));
        (*indices)[i] = malloc(n * sizeof(uint16_t));
        float *values = malloc(n * sizeof(float));
        
        fread((*indices)[i], sizeof(uint16_t), n, f);
        fread(values, sizeof(float), n, f);

        for (uint32_t j = 0; j < n; j++)
        {
            (*dataset)[i][(*indices)[i][j]] = values[j];
        }
        
        // free(indices);
        free(values);
    }
    fclose(f);
}


int main(void){
    int n_images = 9;
    int vec_length = 1024;
    float **dataset;
    uint16_t **indices;
    uint32_t *n_indices;
    read_sparse_dataset("encoded_dataset",
        &dataset, &indices, &n_indices,
        n_images, vec_length);


    
    for (int i = 0; i < n_images; i++)
    {
        printf("%i\n", n_indices[i]);
    }


    for (int j = 0; j < n_indices[8]; j++)
    {
        uint16_t idx = indices[8][j];
        printf("%i: %f\n", idx, dataset[8][idx]);
    }
    
    // for (int i = 0; i < n_images; i++)
    // {
    //     for (int j = 0; j < 1024; j++)
    //     {
    //         printf("%f ", dataset[i][j]);
    //     }
    //     printf("\n\n");
    // }
    



    printf("Success\n");
}