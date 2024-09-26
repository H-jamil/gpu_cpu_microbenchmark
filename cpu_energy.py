import time
import os

def read_cpu_energy():
    """Read the current CPU energy consumption from the RAPL interface."""
    rapl_file_path = '/sys/class/powercap/intel-rapl:0/energy_uj'
    if os.path.exists(rapl_file_path):
        with open(rapl_file_path, 'r') as f:
            energy_uj = int(f.read().strip())  # Energy in microjoules
        return energy_uj / 1e6  # Convert to joules
    else:
        raise FileNotFoundError("Intel RAPL energy file not found.")

def monitor_cpu_energy(duration=10):
    """Monitor and display CPU energy consumption for the given duration."""
    energy_readings = []
    
    # Initial energy reading
    start_energy = read_cpu_energy()
    start_time = time.time()
    
    for i in range(duration):
        time.sleep(1)
        current_energy = read_cpu_energy()
        energy_diff = current_energy - start_energy
        energy_readings.append(energy_diff)
        print(f"Energy used in second {i+1}: {energy_diff:.6f} J")
        start_energy = current_energy

    total_time = time.time() - start_time
    total_energy = sum(energy_readings)
    
    # Print total energy consumption
    print(f"Total energy consumed: {total_energy:.6f} J over {total_time:.2f} seconds")
    print(f"Per second energy consumption: {[f'{e:.6f} J' for e in energy_readings]}")

if __name__ == "__main__":
    monitor_cpu_energy(10)
