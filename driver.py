LATTICE_SIZE = 4
J = 1
T_min = 0.1
T_max = 6
N_temp_step = 20
THROW_AWAY_COUNT = 7000
NUM_SAMPLES = 20000



def main():
    from ising_model_main import write_ising
    from analyze_data import analyze_main
    print('Performing Ising Simulation')
    write_ising()
    print('Analyzing')
    analyze_main()

if __name__ == '__main__':
    main()