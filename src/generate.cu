#include "ant.h"
#include "grid.h"
#include <string>
#include <gsl/gsl_sf_pow_int.h>
#include "possible_states.h"
#include <fstream>
#include <sstream>

__device__
Eigen::Vector2i rotate_i_gpu(Eigen::Vector2i to_be_rotated, Eigen::Vector2i rotater) {
    Eigen::Vector2i retval(to_be_rotated[0] * rotater[0] - to_be_rotated[1] * rotater[1], to_be_rotated[0] * rotater[1] + to_be_rotated[1] * rotater[0]);
    return retval;
}


__device__
Eigen::Vector2i ant_on_patch(Eigen::Vector2i location, int * grid_states, Eigen::Vector2i * state_movements, int x_bound, int y_bound, unsigned int grid_state_start_index, unsigned int state_movement_start_index, unsigned int num_states) {
    
    unsigned int index = grid_state_start_index + (location[0] + x_bound) * (y_bound * 2 + 1) + (location[1] + y_bound);
    auto ant_movement = state_movements[state_movement_start_index + grid_states[index] % num_states];
    grid_states[index]++;

    return ant_movement; 
}

__device__
void ant_move(Eigen::Vector2i ant_relative_direction, Eigen::Vector2i &ant_location, Eigen::Vector2i &ant_orientation, int x_bound, int y_bound) {

    assert(ant_relative_direction[0] == 0 || ant_relative_direction[1] == 0);
    assert(abs(ant_relative_direction[0]) < x_bound && abs(ant_relative_direction[1]) < y_bound);

    Eigen::Vector2i map_relative_direction = rotate_i_gpu(ant_relative_direction, ant_orientation);
    ant_location += map_relative_direction;
    
    if(ant_location[0] > x_bound) {
            ant_location[0] = -x_bound + (ant_location[0] - x_bound) - 1;
    }
    else if(ant_location[0] < -x_bound) {
        ant_location[0] = x_bound + (ant_location[0] + x_bound) + 1;
    }
    else if(ant_location[1] > y_bound) {
        ant_location[1] = -y_bound + (ant_location[1] - y_bound) - 1;
    }
    else if(ant_location[1] < -y_bound) {
        ant_location[1] = y_bound + (ant_location[1] + y_bound) + 1;
    }

    //map_relative_direction.norm() crashes so....
   
    int magnitude = sqrt((double) (map_relative_direction[0] * map_relative_direction[0] + map_relative_direction[1] * map_relative_direction[1])); 
 
    ant_orientation = map_relative_direction / magnitude;
}
__global__
void run_ant_gpu(Eigen::Vector2i ant_location, Eigen::Vector2i ant_orientation, int * grid_states, Eigen::Vector2i * state_movements, int x_bound, int y_bound, unsigned int iteration_count, unsigned int num_states, int num_grid_states) {

    unsigned int grid_state_start_index, state_movement_start_index;

    unsigned int thread_index = blockIdx.x*blockDim.x + threadIdx.x;

    grid_state_start_index = (x_bound * 2 + 1) * (y_bound * 2 + 1) * thread_index;

    if(grid_state_start_index >= num_grid_states) {
        //printf("Thread: %d is operating on unaddressed memory.\n", thread_index);
        return;
    }

    state_movement_start_index = num_states * thread_index;

    for(unsigned int i = 0; i < iteration_count; i++) {
        auto movement = ant_on_patch(ant_location, grid_states, state_movements, x_bound, y_bound, grid_state_start_index, state_movement_start_index, num_states);
        ant_move(movement, ant_location, ant_orientation, x_bound, y_bound);
    }
} 

void run_ant_main(grid::grid &squares, ant::ant_interface * ants, unsigned int num_ants);

void write_binary_blob(std::string file_dir, int * grid_states, Eigen::Vector2i * state_movements, int x_bound, int y_bound, int initial_grid_index, int initial_state_index, int num_states) {
    
    std::stringstream file_name;
    
    file_name << file_dir << "/generated_data";

    for(int i = 0; i < num_states; i++) {
        file_name << "__" << state_movements[initial_state_index + i][0] << "_" << state_movements[initial_state_index + i][1];
    }

    file_name << ".antgen";

    std::ofstream out_file;

    out_file.open(file_name.str(), std::ios::out | std::ios::binary);

    int x_size = x_bound * 2 + 1;
    int y_size = y_bound * 2 + 1; 

    char size_data[2];

    std::memcpy(size_data, &x_size, 2);
    out_file.write(size_data, 2);
    std::memcpy(size_data, &y_size, 2);
    out_file.write(size_data, 2);

    int total_grid_size = x_size * y_size;

    for(int i = 0; i < total_grid_size; i++) {
        char state_index_data[1];
        
        uint8_t state_index = static_cast<uint8_t>(grid_states[initial_grid_index + i] % num_states);
 
        std::memcpy(state_index_data, &state_index, 1);

        out_file.write(state_index_data, 1);
    }
    out_file.close();
}

void generate_and_run(int minimum_mov, int maximum_mov, int x_bound, int y_bound, int num_states, bool is_forward_back_allowed, bool enforce_symmetric, int num_iterations, std::string file_dir, int max_combinations, int initial_num, int batch_size, std::vector<std::vector<int>> &states) {

    if(max_combinations < 0) {
        max_combinations = gsl_sf_pow_int((maximum_mov - minimum_mov + 1) * 2 * (is_forward_back_allowed ? 2 : 1), num_states); //Multiplied by 2 for sign bit
    }

    assert(max_combinations >= 0); //Overflow

    unsigned int divisor = is_forward_back_allowed ? gsl_sf_pow_int(2, num_states) : 1; 

    std::vector<std::vector<int>> forward_back_sets;

    if(is_forward_back_allowed) {
        generate_all_states(forward_back_sets, 0, 1, num_states);
    
        assert(forward_back_sets.size() == divisor);
    }

    std::cout << "max_combinations: " << max_combinations << std::endl;

    int epoch_num = 0;

    for(int generation_num = initial_num; generation_num < max_combinations;) {

        std::cout << "Epoch: " << epoch_num << std::endl;        

        epoch_num++;
    
        int remaining = max_combinations - generation_num;

        int num_in_this_batch = remaining < batch_size ? remaining : batch_size; 

        Eigen::Vector2i ant_location(0, 0); //We assume for now all ants start at 0, 0
        Eigen::Vector2i ant_orientation(0, 1); //And facing up

        int num_grid_states = num_in_this_batch * (x_bound * 2 + 1) * (y_bound * 2 + 1);
        int * grid_states_host = new int [num_grid_states];

        int num_state_movements = num_in_this_batch * num_states * (enforce_symmetric ? 2 : 1);
        Eigen::Vector2i * state_movements_host = new Eigen::Vector2i [num_state_movements];

        for(int i = 0; i < num_grid_states; i++) {
            grid_states_host[i] = 0;
        }
        
        int overall_index = 0;

        for(int i = 0; i < num_in_this_batch; i++) {
            int state_index = 0;
            for(auto state : states[generation_num / divisor]) { //Delibrate truncation
                int index = is_forward_back_allowed ? forward_back_sets[generation_num % divisor][state_index] : 1;
            
                state_index++;

                state_movements_host[overall_index][0] = 0;
                state_movements_host[overall_index][1] = 0;

                state_movements_host[overall_index][index] = state;

                overall_index++;
            }
            generation_num++;
        }
        
        assert(overall_index == num_state_movements);

        std::cout << "Copy Checkpoint 1" << std::endl;

        int * grid_states_gpu;
        Eigen::Vector2i * state_movements_gpu;

        cudaMalloc(&grid_states_gpu, num_grid_states * sizeof(int));
        cudaMalloc(&state_movements_gpu, num_state_movements * sizeof(Eigen::Vector2i));

        cudaMemcpy(grid_states_gpu, grid_states_host, num_grid_states * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(state_movements_gpu, state_movements_host, num_state_movements * sizeof(Eigen::Vector2i), cudaMemcpyHostToDevice);
        
        std::cout << "Copy Checkpoint 2" << std::endl;

        run_ant_gpu<<<(num_in_this_batch + 255) / 256, 256>>>(ant_location, ant_orientation, grid_states_gpu, state_movements_gpu, x_bound, y_bound, num_iterations, num_states, num_grid_states);

        cudaDeviceSynchronize();

        std::cout << "After run" << std::endl;

        cudaMemcpy(grid_states_host, grid_states_gpu, num_grid_states * sizeof(int), cudaMemcpyDeviceToHost);
      
        cudaFree(grid_states_gpu);
        cudaFree(state_movements_gpu);

        for(int i = 0; i < num_in_this_batch; i++) {
            write_binary_blob(file_dir, grid_states_host, state_movements_host, x_bound, y_bound, i * (x_bound * 2 + 1) * (y_bound * 2 + 1) , i * num_states, num_states);
        }
        
        delete state_movements_host; 
        delete grid_states_host;
    }
}

int main(int argc, char** argv) {

    //Actual Problem: (./bin/generator 3000 data 300000 1 4 25 25 6 0 0) 
    
    //Trivial Test: (./bin/app 2000) 

    int minimum_mov = 1;
    int maximum_mov = 1;
    int x_bound = 10;
    int y_bound = 10;
    int num_states = 2;
    bool is_forward_back_allowed = false;
    bool enforce_symmetric = true;
    int num_iterations = 1000;
    std::string file_dir = "data";
    int max_combinations = -1;
    int initial_num = 0;
    int batch_size = 8;        

    if(argc > 4) {
        minimum_mov = std::stoi(argv[4]);
    }

    if(argc > 5) {
        maximum_mov = std::stoi(argv[5]);
    }

    if(argc > 6) {
        x_bound = std::stoi(argv[6]);
    }

    if(argc > 7) {
        y_bound = std::stoi(argv[7]);
    }

    if(argc > 8) {
        num_states = std::stoi(argv[8]);
    }

    if(argc > 9) {
        is_forward_back_allowed = std::stoi(argv[9]);
    }

    if(argc > 10) {
        enforce_symmetric = std::stoi(argv[10]);
    }

    if(argc > 1) {
        num_iterations = std::stoi(argv[1]);
    }

    if(argc > 2) {
        file_dir = argv[2];
    }

    if(argc > 11) {
        max_combinations = std::stoi(argv[11]);
    }

    if(argc > 12) {
        initial_num = std::stoi(argv[12]);
    }

    if(argc > 3) {
        batch_size = std::stoi(argv[3]);
    }        

    std::vector<std::vector<int>>  states;

    generate_all_states(states, minimum_mov, maximum_mov, num_states);

    generate_and_run(minimum_mov, maximum_mov, x_bound, y_bound, num_states, is_forward_back_allowed, enforce_symmetric, num_iterations, file_dir, max_combinations, initial_num, batch_size, states);
   
    return 0; 
}
