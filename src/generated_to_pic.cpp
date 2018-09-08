#include "display.h"
#include "hsv_to_rgb.h"
#include <fstream>

void get_binary_blob(std::string file_name, grid::grid &out_grid) {

    std::ifstream in_file;

    in_file.open(file_name, std::ios::in | std::ios::binary);

    uint16_t x_size;
    uint16_t y_size;

    char size_data[2];

    in_file.read(size_data, 2);
    std::memcpy(&x_size, size_data, 2);
    in_file.read(size_data, 2);
    std::memcpy(&y_size, size_data, 2);

    std::vector<grid::state> temp_states;
    grid::grid temp_grid((x_size - 1) / 2, (y_size - 1) / 2, temp_states);


    int max_state = 0;
    for(auto &column : temp_grid.patches) {
        for(auto &this_patch : column) {
            char state_index[1];

            in_file.read(state_index, 1);

            this_patch.state_index = 0;

            std::memcpy(&this_patch.state_index, state_index, 1);

            max_state = this_patch.state_index > max_state ? this_patch.state_index : max_state;
        }
    }

    double saturation = 1.0;
    double value = 1.0;

    for(int i = 0; i <= max_state; i++) {
        hsv this_hsv = {360.0 * i / ((double) max_state), saturation, value};

        auto color_vals = hsv2rgb(this_hsv);

        Eigen::Vector3d color_vec(color_vals.r, color_vals.b, color_vals.g);

        grid::state this_state(color_vec, Eigen::Vector2i(0, 0));

        temp_states.push_back(this_state);
    }
    
    in_file.close();

    temp_grid.states = temp_states;

    out_grid = temp_grid;
    
}

std::vector<grid::state> temp_states;

grid::grid grid_to_display(0, 0, temp_states);

ant::base_ant only_ant(1, 1);

ant::base_ant ants[1] = {only_ant};

void display_generated_grid() {
    display(grid_to_display, ants, 1);
}

int main(int argc, char** argv) {
    if(argc == 1) {
        std::cout << "File name required." << std::endl;
        return 1;
    }
    
    std::string file_name = argv[1];

    get_binary_blob(file_name, grid_to_display);

    glutInit(&argc, argv);                 // Initialize GLUT
    glutCreateWindow("OpenGL Setup Test"); // Create a window with the given title
    glutInitWindowSize(320, 320);   // Set the window's initial width & height
    glutInitWindowPosition(50, 50); // Position the window's initial top-left corner

    glutDisplayFunc(display_generated_grid);
    glutMainLoop(); 
    
    return 0;    
}
