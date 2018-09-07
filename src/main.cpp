#include <GL/glut.h>  // GLUT, include glu.h and gl.h
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "ant.h"
#include "grid.h"
#include <atomic>
#include <thread>
#include <chrono>
#include "hsv_to_rgb.h"

void run_ant_main(grid::grid &squares, ant::ant_interface * ants, unsigned int num_ants);

/* Handler for window-repaint event. Call back when the window first appears and
   whenever the window needs to be re-painted. */
void display(grid::grid &squares, ant::ant_interface * ants, unsigned int num_ants) {
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
   
    double x_dim = squares.patches.size();
    double y_dim = squares.patches[0].size();

    double x_scale = 1.0 / (x_dim);
    double y_scale = 1.0 / (y_dim);

    for(auto &column : squares.patches) {
        for(auto &this_patch : column) {
            if(this_patch.state_index == 0) {continue;}
            
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

std::random_device rd;
std::default_random_engine engine{rd()};

//Actual Problem: (./bin/app 0 1000 10000) //TODO: VALIDATE

//std::uniform_int_distribution<int> mov_dist(1, 4);
//std::uniform_int_distribution<int> sign_dist(0, 1);
//
//int x_bound = 50;
//int y_bound = 50;
//
//std::vector<Eigen::Vector2i> movement_states = {
//    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
//    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
//    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
//    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
//    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
//    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
//};

//Trivial Test: (./bin/app 0 100 2000) 

std::uniform_int_distribution<int> mov_dist(1, 1);
std::uniform_int_distribution<int> sign_dist(0, 1);

int x_bound = 10;
int y_bound = 10;

std::vector<Eigen::Vector2i> movement_states = {
    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
    {0, mov_dist(engine) * (sign_dist(engine) * 2 - 1)},
};

std::vector<grid::state> grid_states;

grid::grid main_grid(x_bound, y_bound, grid_states);

ant::base_ant only_ant(x_bound, y_bound); 

ant::base_ant ants[1] = {only_ant};

int run_main_per_loop = 100;

int sleep_per_loop = 0;

int num_iterations = -1;

void display_ant() {
    
    static int k = 0;
   
    if(num_iterations < 0 || k < num_iterations) {
        for(int i = 0; i < run_main_per_loop; i++) {
            run_ant_main(main_grid, ants, 1);
            k++;
        }
    }

    display(main_grid, ants, 1);

    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_per_loop));
}


/* Main function: GLUT runs as a console application starting at main()  */
int main(int argc, char** argv) {

    if(argc > 1) {
        sleep_per_loop = std::stoi(argv[1]);
    }
    
    if(argc > 2) {
        run_main_per_loop = std::stoi(argv[2]);
    }

    if(argc > 3) {
        num_iterations = std::stoi(argv[3]);
    }

    glutInit(&argc, argv);                 // Initialize GLUT
    glutCreateWindow("OpenGL Setup Test"); // Create a window with the given title
    glutInitWindowSize(320, 320);   // Set the window's initial width & height
    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner

    std::atomic<bool> exit(false);

    double saturation = 1.0;
    double value = 1.0;

    for(int i = (movement_states.size() - 1); i >= 0; i--) {
        auto this_state = movement_states[i]; 

        movement_states.push_back(this_state);
    }

    for(size_t i = 0; i < movement_states.size(); i++) {
        hsv this_hsv = {360.0 * i / ((double) movement_states.size() - 1), saturation, value};

        auto color_vals = hsv2rgb(this_hsv);

        Eigen::Vector3d color_vec(color_vals.r, color_vals.b, color_vals.g);

        grid::state this_state(color_vec, movement_states[i]);       
 
        grid_states.push_back(this_state);
    }

    main_grid = grid::grid(x_bound, y_bound, grid_states);

    glutIdleFunc(display_ant);
    
    glutMainLoop();           // Enter the event-processing loop

    exit.store(true);

    return 0;
}
