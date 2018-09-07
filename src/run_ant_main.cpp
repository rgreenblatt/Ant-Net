#include "ant.h"
#include "grid.h"

void run_ant_main(grid::grid &squares, ant::ant_interface * ants, unsigned int num_ants) {
    
    for(int i = 0; i < num_ants; i++) {
        auto &ant = ants[i];
        auto ant_relative_direction = squares.ant_on_patch(ant.get_location());
        
        ant.move(ant_relative_direction);
    }
}
