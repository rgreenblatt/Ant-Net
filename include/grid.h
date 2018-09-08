#pragma once

#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>

namespace grid {
    struct patch {
        int state_index = 0;
        Eigen::Vector2i location; 
        
        patch(int state_index_in, Eigen::Vector2i location_in) :
        state_index(state_index_in),
        location(location_in)
        {}
    };
    
    struct state {
        Eigen::Vector3d color;
        Eigen::Vector2i ant_relative_direction;
        
        state(Eigen::Vector3d color_in, Eigen::Vector2i ant_relative_direction_in) :
        color(color_in),
        ant_relative_direction(ant_relative_direction_in)
        {} 
        
        state(Eigen::Vector2i ant_relative_direction_in) :
        ant_relative_direction(ant_relative_direction_in)
        {
            color = Eigen::Vector3d(0, 0, 0);
        } 

    };

    class grid {
    public: 
        grid(int x_bound, int y_bound, std::vector<state> states_in) :
        states(states_in)
        {
            for(int x = -x_bound; x <= x_bound; x++) {
                std::vector<patch> column;
                for(int y = -y_bound; y <= y_bound; y++) {
                    Eigen::Vector2i location(x, y); 
                    patch this_patch(0, location);

                    column.push_back(this_patch);
                }
                
                patches.push_back(std::move(column));
            }
        }

        Eigen::Vector2i ant_on_patch(Eigen::Vector2i location) {
            auto &column = patches[location[0] + patches.size() / 2];
            auto &this_patch = column[location[1] + column.size() / 2];

            auto ant_movement = states[this_patch.state_index % states.size()].ant_relative_direction;
            this_patch.state_index++;

            return ant_movement;
        }

        std::vector<std::vector<patch>> patches; 
        std::vector<state> states;
    };
};
