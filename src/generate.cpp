#include "ant.h"
#include "grid.h"
#include <string>
#include <gsl/gsl_sf_pow_int.h>
#include "possible_states.h"
#include <fstream>
#include <sstream>

void run_ant_main(grid::grid &squares, ant::ant_interface * ants, unsigned int num_ants);

void write_binary_blob(std::string file_dir, const grid::grid &log_grid) {
    
    std::stringstream file_name;
    
    file_name << file_dir << "/generated_data";

    for(const auto &this_state : log_grid.states) {
        file_name << "__" << this_state.ant_relative_direction[0] << "_" << this_state.ant_relative_direction[1];
    }

    file_name << ".antgen";

    std::ofstream out_file;

    out_file.open(file_name.str(), std::ios::out | std::ios::binary);

    size_t x_size = log_grid.patches.size();
    size_t y_size = log_grid.patches[0].size();

    char size_data[2];

    std::memcpy(size_data, &x_size, 2);
    out_file.write(size_data, 2);
    std::memcpy(size_data, &y_size, 2);
    out_file.write(size_data, 2);

    for(auto &column : log_grid.patches) {
        for(auto &this_patch : column) {
            char state_index_data[1];
           
            uint8_t state_index = static_cast<uint8_t>(this_patch.state_index % log_grid.states.size());
 
            std::memcpy(state_index_data, &state_index, 1);

            out_file.write(state_index_data, 1);
        }
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

    for(int generation_num = initial_num; generation_num < max_combinations;) {

        int remaining = max_combinations - generation_num;

        int num_in_this_batch = remaining < batch_size ? remaining : batch_size; 

        std::vector<grid::grid> grids; //Grids are by batch so we can write their values
        std::vector<ant::base_ant> all_ants;

        for(int i = 0; i < num_in_this_batch; i++) {
            
            std::vector<grid::state> grid_states;                
            ant::base_ant only_ant(x_bound, y_bound);

            all_ants.push_back(only_ant);

            int state_index = 0;

            for(auto state : states[generation_num / divisor]) { //Delibrate truncation
                Eigen::Vector2i state_vec(0, 0);

                int index = is_forward_back_allowed ? forward_back_sets[generation_num % divisor][state_index] : 1;

                state_vec[index] = state;
            
                grid_states.push_back(state_vec);

                state_index++;
            }

            grids.push_back(grid::grid(x_bound, y_bound, grid_states));

            generation_num++;
        }

        //TODO: Optimize
        
        for(int i = 0; i < num_in_this_batch; i++) {
            for(int k = 0; k < num_iterations; k++) {
                run_ant_main(grids[i], &all_ants[i], 1); 
            }
        }
        
        for(auto &log_grid : grids) {
            write_binary_blob(file_dir, log_grid);

        }
    }
}

int main(int argc, char** argv) {

    //Actual Problem: (./bin/app 10000) 
    
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
