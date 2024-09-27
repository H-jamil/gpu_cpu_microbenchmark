#include "energy_monitor.h"
#include <string.h>
#include <time.h>

double cpu_energy = 0;
double gpu_energy[2] = {0, 0};
int stop_monitoring = 0;

void *monitor_cpu_energy(void *arg) {
    const char *rapl_file_path = "/sys/class/powercap/intel-rapl:0/energy_uj";
    FILE *rapl_file;
    unsigned long long prev_energy = 0, curr_energy;

    while (!stop_monitoring) {
        rapl_file = fopen(rapl_file_path, "r");
        if (rapl_file) {
            fscanf(rapl_file, "%llu", &curr_energy);
            fclose(rapl_file);

            if (prev_energy > 0) {
                cpu_energy += (curr_energy - prev_energy) / 1e6; // Convert to joules
            }
            prev_energy = curr_energy;
        }
        sleep(1);
    }
    return NULL;
}

void *monitor_gpu_energy(void *arg) {
    FILE *fp;
    char line[1024];
    float power_draw[2];
    struct timespec start, end;
    double elapsed_time;

    while (!stop_monitoring) {
        clock_gettime(CLOCK_MONOTONIC, &start);

        fp = popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits", "r");
        if (fp == NULL) {
            printf("Failed to run nvidia-smi command\n");
            return NULL;
        }

        int gpu_index = 0;
        while (fgets(line, sizeof(line), fp) != NULL) {
            if (gpu_index < 2) {
                sscanf(line, "%f", &power_draw[gpu_index]);
                gpu_index++;
            }
        }

        pclose(fp);

        clock_gettime(CLOCK_MONOTONIC, &end);
        elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        for (int i = 0; i < 2; i++) {
            gpu_energy[i] += power_draw[i] * elapsed_time; // Energy = Power * Time
        }

        // Sleep for the remaining time to make it a full second
        if (elapsed_time < 1.0) {
            usleep((1.0 - elapsed_time) * 1e6);
        }
    }
    return NULL;
}

void start_energy_monitoring() {
    pthread_t cpu_thread, gpu_thread;
    stop_monitoring = 0;

    if (pthread_create(&cpu_thread, NULL, monitor_cpu_energy, NULL) != 0) {
        fprintf(stderr, "Failed to create CPU monitoring thread\n");
        return;
    }

    if (pthread_create(&gpu_thread, NULL, monitor_gpu_energy, NULL) != 0) {
        fprintf(stderr, "Failed to create GPU monitoring thread\n");
        return;
    }
}

void stop_energy_monitoring() {
    stop_monitoring = 1;
    sleep(1); // Give threads time to finish
}