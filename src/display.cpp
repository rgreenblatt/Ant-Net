#include "display.h"



void display(grid::grid &squares, ant::ant_interface * ants, unsigned int num_ants) {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
   
    double x_dim = squares.patches.size();
    double y_dim = squares.patches[0].size();

    double x_scale = 1.0 / (x_dim);
    double y_scale = 1.0 / (y_dim);

    for(auto &column : squares.patches) {
        for(auto &this_patch : column) {
            if(this_patch.state_index % squares.states.size() == 0) {continue;}
            
            glBegin(GL_QUADS);
                
                
                auto &this_state_color = squares.states[this_patch.state_index % squares.states.size()].color;

                glColor3d(this_state_color[0], this_state_color[1], this_state_color[2]);

                glVertex2d((2 * this_patch.location[0] + 1.0) * x_scale, (2 * this_patch.location[1] + 1.0) * y_scale);
                glVertex2d((2 * this_patch.location[0] - 1.0) * x_scale, (2 * this_patch.location[1] + 1.0) * y_scale);
                glVertex2d((2 * this_patch.location[0] - 1.0) * x_scale, (2 * this_patch.location[1] - 1.0) * y_scale);
                glVertex2d((2 * this_patch.location[0] + 1.0) * x_scale, (2 * this_patch.location[1] - 1.0) * y_scale);
            glEnd();
        }
    }
 
    glBegin(GL_LINES);
        glColor3d(0.0, 0.0, 0.0);
        
        for(int i = 1; i < 2*x_dim + 1; i++) {
            double x_pos = (2.0 * i) * x_scale - 1.0;
            glVertex2d(x_pos, -1.0);
            glVertex2d(x_pos, 1.0);
        }
         
        for(int i = 1; i < 2*y_dim +1 ; i++) {
            double y_pos = (2.0 * i) * y_scale - 1.0;
            glVertex2d(-1.0, y_pos);
            glVertex2d(1.0, y_pos);
        }
    glEnd();

    for(unsigned int i = 0; i < num_ants; i++) {
        glBegin(GL_TRIANGLES);
            auto &ant = ants[i];

            auto &this_ant_color = ant::ant_interface::ant_color_state[ant.get_state() % ant::ant_interface::ant_color_state.size()];

            glColor3d(this_ant_color[0], this_ant_color[1], this_ant_color[2]);

            Eigen::Vector2d tip(x_scale, 0);
            Eigen::Vector2d top(-x_scale, y_scale);
            Eigen::Vector2d bot(-x_scale, -y_scale);

            Eigen::Vector2d rotated_tip = rotate_d(tip, ant.get_orientation());
            Eigen::Vector2d rotated_top = rotate_d(top, ant.get_orientation());
            Eigen::Vector2d rotated_bot = rotate_d(bot, ant.get_orientation());

            Eigen::Vector2d location = ant.get_location().cast<double>();

            location[0] *= 2 * x_scale;
            location[1] *= 2 * y_scale;

            Eigen::Vector2d final_tip = rotated_tip + location;
            Eigen::Vector2d final_top = rotated_top + location;           
            Eigen::Vector2d final_bot = rotated_bot + location;

            glVertex2d(final_tip[0], final_tip[1]);
            glVertex2d(final_top[0], final_top[1]);
            glVertex2d(final_bot[0], final_bot[1]);
        
        glEnd(); 
    }   
    
    glFlush();
}
