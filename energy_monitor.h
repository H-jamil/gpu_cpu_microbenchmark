#ifndef ENERGY_MONITOR_H
#define ENERGY_MONITOR_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>

extern double cpu_energy;
extern double gpu_energy[2];
extern int stop_monitoring;

void *monitor_cpu_energy(void *arg);
void *monitor_gpu_energy(void *arg);
void start_energy_monitoring();
void stop_energy_monitoring();

#endif