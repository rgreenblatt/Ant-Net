#include "ant.h"
#include <assert.h>
#include <iostream>

namespace ant {
    void base_ant::move(Eigen::Vector2i ant_relative_direction) {

        assert(ant_relative_direction[0] == 0 || ant_relative_direction[1] == 0);

        assert(abs(ant_relative_direction[0]) < x_bound && abs(ant_relative_direction[1]) < y_bound);

        Eigen::Vector2i map_relative_direction = rotate_i(ant_relative_direction, orientation);
        
        location += map_relative_direction;

        if(location[0] > x_bound) {
            location[0] = -x_bound + (location[0] - x_bound) - 1;
        }
        else if(location[0] < -x_bound) {
            location[0] = x_bound + (location[0] + x_bound) + 1;
        }
        else if(location[1] > y_bound) {
            location[1] = -y_bound + (location[1] - y_bound) - 1;
        }
        else if(location[1] < -y_bound) {
            location[1] = y_bound + (location[1] + y_bound) + 1;
        }

        orientation = map_relative_direction / map_relative_direction.norm();
    }

    std::vector<Eigen::Vector3d> ant_interface::ant_color_state = {{0.0, 0.0, 0.0}}; 
}
