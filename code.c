#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_FILE_SIZE 500000000 // Definimos buffer de 500MB para poder leer ficheros grandes

// Estructura para facilitar el ordenamiento de los vecinos
typedef struct
{
    int index;   // Índice de la fila (día)
    double dist; // Distancia Euclídea respecto al día objetivo
} Vecino;

// Función comparadora para ordenar de menor a mayor distancia
int compareVecinos(const void *a, const void *b)
{
    Vecino *v1 = (Vecino *)a;
    Vecino *v2 = (Vecino *)b;
    if (v1->dist > v2->dist) return 1;
    if (v1->dist < v2->dist) return -1;
    return 0;
}

int main(int argc, char *argv[])
{
    int pid, prn, filas, columnas, k, nH;
    char *filename;
    double *datos = NULL;

    // 1. INICIALIZACIÓN DEL ENTORNO MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &prn);

    // Comprobación de número de argumentos
    if (argc != 5)
    {
        if (pid == 0)
            printf("Uso: ./prog K fichero nP nH\n");
        MPI_Finalize();
        return 1;
    }

    // Lectura de parámetros de consola
    k = atoi(argv[1]);  // Vecinos
    filename = argv[2];  // Archivo
    prn = atoi(argv[3]); // Procesos
    nH = atoi(argv[4]);  // Hilos

    // Configuramos OpenMP con el número de hilos solicitado
    omp_set_num_threads(nH);

    // 2. LECTURA DE DATOS CON MPI_FILE
    if (pid == 0)
    {
        MPI_File fh;
        MPI_Status status;
        
        // Abrimos el fichero en modo solo lectura
        int err = MPI_File_open(MPI_COMM_SELF, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS)
        {
            printf("Error abriendo el archivo con MPI_IO: %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Reservamos memoria para el buffer gigante de lectura
        char *file_buffer = (char *)malloc(MAX_FILE_SIZE * sizeof(char));

        // Leemos el fichero completo (binario) al buffer de memoria
        MPI_File_read(fh, file_buffer, MAX_FILE_SIZE, MPI_CHAR, &status);
        
        // Verificamos cuántos bytes hemos leído realmente para poner el final de cadena
        int count;
        MPI_Get_count(&status, MPI_CHAR, &count);
        file_buffer[count] = '\0';

        MPI_File_close(&fh); // Cerramos el fichero, ya lo tenemos en RAM

        // Usamos strtok para trocear el texto por espacios, comas o saltos de línea
        // Leemos filas y columnas
        char *token = strtok(file_buffer, " \n\r\t,");
        if(token != NULL) filas = atoi(token);

        token = strtok(NULL, " \n\r\t,");
        if(token != NULL) columnas = atoi(token);

        // Reservamos la matriz de datos
        datos = (double *)malloc(filas * columnas * sizeof(double));
        
        // Convertimos el resto del texto a números double
        for (int i = 0; i < filas * columnas; i++)
        {
            token = strtok(NULL, " \n\r\t,");
            if (token != NULL)
                datos[i] = atof(token);
            else
                datos[i] = 0.0; // Protección contra ficheros corruptos
        }
        
        free(file_buffer); // Liberamos el buffer de texto crudo
    }

    // 3. DISTRIBUCIÓN DE DATOS (BROADCAST)
    // El maestro envía las dimensiones a cada proceso
    MPI_Bcast(&filas, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&columnas, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Los esclavos reservan su memoria para recibir la matriz
    if (pid != 0)
    {
        datos = (double *)malloc(filas * columnas * sizeof(double));
    }
    
    // El maestro envía la matriz completa a todos
    MPI_Bcast(datos, filas * columnas, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *mapes_resultados = NULL;
    double *predicciones_resultados = NULL;

    int num_predicciones = (filas > 1000) ? 1000 : (filas - 1);
    int colummna_prediccion = filas - num_predicciones;

    if (pid == 0)
    {
        mapes_resultados = (double *)malloc(num_predicciones * sizeof(double));
        predicciones_resultados = (double *)malloc(num_predicciones * columnas * sizeof(double));
    }

    // Sincronización para medición del tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // 4. BUCLE PRINCIPAL DE PREDICCIÓN
    for (int i = 0; i < num_predicciones; i++)
    {
        int diaObjetivo = colummna_prediccion + i;
        int historia_disponible = diaObjetivo; // Solo podemos mirar el pasado

        // Estructura local para cada proceso
        Vecino *mis_candidatos = (Vecino *)malloc(historia_disponible * sizeof(Vecino));

        // Inicio zona paralelizada con OpenMP
        #pragma omp parallel
        {
            double dist, diff;

            #pragma omp for schedule(static) nowait
            for (int f = pid; f < historia_disponible; f += prn)
            {
                if (f >= filas - 1) continue;

                // Cálculo Distancia Euclídea
                dist = 0;
                for (int c = 0; c < columnas; c++)
                {
                    diff = datos[diaObjetivo * columnas + c] - datos[f * columnas + c];
                    dist += diff * diff;
                }
                mis_candidatos[f].index = f;
                mis_candidatos[f].dist = dist;
            }
        }

        double *distancias_locales = (double *)calloc(historia_disponible, sizeof(double));
        for (int f = pid; f < historia_disponible; f += prn)
        {
            distancias_locales[f] = mis_candidatos[f].dist;
        }

        double *todas_distancias = NULL;
        if (pid == 0)
            todas_distancias = (double *)malloc(historia_disponible * sizeof(double));

        // Sumamos los arrays de todos los procesos
        MPI_Reduce(distancias_locales, todas_distancias, historia_disponible, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // 5. LÓGICA DEL MAESTRO
        if (pid == 0)
        {
            // Reconstruir lista de vecinos completa
            Vecino *todos = (Vecino *)malloc(historia_disponible * sizeof(Vecino));
            int count = 0;
            for (int j = 0; j < historia_disponible; j++)
            {
                todos[count].index = j;
                todos[count].dist = todas_distancias[j];
                count++;
            }

            // Ordenar por menor distancia
            qsort(todos, count, sizeof(Vecino), compareVecinos);

            // Predecir con la media de los k vecinos más cercanos
            double *prediccion = (double *)calloc(columnas, sizeof(double));
            for (int v = 0; v < k; v++)
            {
                int idx_siguiente = todos[v].index + 1; // Miramos lo que pasó el día SIGUIENTE
                for (int c = 0; c < columnas; c++)
                {
                    prediccion[c] += datos[idx_siguiente * columnas + c];
                }
            }

            // Calcular MAPE
            double error_dia = 0;
            for (int c = 0; c < columnas; c++)
            {
                prediccion[c] /= k;
                
                predicciones_resultados[i * columnas + c] = prediccion[c];

                double real = datos[diaObjetivo * columnas + c];
                if (real != 0)
                    error_dia += fabs((real - prediccion[c]) / real);
            }
            mapes_resultados[i] = (error_dia / columnas) * 100.0;

            free(todos);
            free(prediccion);
        }

        // Limpieza
        free(mis_candidatos);
        free(distancias_locales);
        if (pid == 0) free(todas_distancias);
    }

    double end_time = MPI_Wtime();

    // 6. ESCRITURA DE FICHEROS CON MPI_FILE
    if (pid == 0)
    {
        printf("--- Iniciando escritura de resultados ---\n");

        // 6.1. Escritura de Predicciones.txt
        MPI_File fhPred;
        MPI_File_open(MPI_COMM_SELF, "Predicciones.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fhPred);
        
        char *linea_buffer = (char *)malloc(columnas * 30 * sizeof(char)); 
        for (int i = 0; i < num_predicciones; i++)
        {
            int offset = 0;
            for (int c = 0; c < columnas; c++)
            {
                int idx = i * columnas + c;
                offset += sprintf(linea_buffer + offset, "%.4f ", predicciones_resultados[idx]);
            }
            offset += sprintf(linea_buffer + offset, "\n");
            MPI_File_write(fhPred, linea_buffer, offset, MPI_CHAR, MPI_STATUS_IGNORE);
        }
        free(linea_buffer);
        MPI_File_close(&fhPred);
        printf("Predicciones.txt escrito correctamente.\n");

        // 6.2. Escritura de MAPE.txt
        MPI_File fhMape;
        MPI_File_open(MPI_COMM_SELF, "MAPE.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fhMape);
        
        char mape_buffer[64];
        double mape_global = 0.0;
        for (int i = 0; i < num_predicciones; i++)
        {
            int len = sprintf(mape_buffer, "%.4f\n", mapes_resultados[i]);
            MPI_File_write(fhMape, mape_buffer, len, MPI_CHAR, MPI_STATUS_IGNORE);
            mape_global += mapes_resultados[i];
        }
        MPI_File_close(&fhMape);
        printf("MAPE.txt escrito correctamente.\n");

        if (num_predicciones > 0) mape_global /= num_predicciones;

        // 6.3. Escritura de Tiempo.txt
        MPI_File fhTiempo;
        // MPI_MODE_APPEND añade al final sin borrar lo anterior
        MPI_File_open(MPI_COMM_SELF, "Tiempo.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &fhTiempo);

        char tiempo_buffer[512];
        int lenT = sprintf(tiempo_buffer, "Tiempo: %.6f s | Fichero: %s | MAPE Global: %.4f %% | Procesos: %d | Hilos: %d\n",
                end_time - start_time, filename, mape_global, prn, nH);
        
        MPI_File_write(fhTiempo, tiempo_buffer, lenT, MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&fhTiempo);
        printf("Tiempo.txt escrito correctamente.\n");

        free(mapes_resultados);
        free(predicciones_resultados);
    }
    
    // Limpieza y fin del entorno MPI
    free(datos);
    MPI_Finalize();
    return 0;
}