#include "grid.h"
#include "ant.h"
#include <gtest/gtest.h>

TEST(ant, move) {
    
    std::default_random_engine engine{static_cast<unsigned int>(testing::UnitTest::GetInstance()->random_seed())};

    std::uniform_int_distribution<int> dist(10, 1000);

    int x_bound = dist(engine);
    int y_bound = dist(engine);

    ant::base_ant test_ant(x_bound, y_bound, true);

    auto init_loc = test_ant.get_location(); 
    auto init_orient = test_ant.get_orientation(); 

    std::uniform_int_distribution<int> x_or_y_dist(0, 1);
    std::uniform_int_distribution<int> sign_dist(0, 1);    

    Eigen::Vector2i movement(0, 0);
    
    std::uniform_int_distribution<int> movement_dist(1, 5);
    std::uniform_int_distribution<int> direction_movement_dist(0, 1);

    movement[direction_movement_dist(engine)] = movement_dist(engine) * (2 * sign_dist(engine) - 1);

    Eigen::Vector2i expected_loc = init_loc + rotate_i(movement, init_orient);

    if(expected_loc[0] > x_bound) {
        expected_loc[0] = -x_bound + (expected_loc[0] - x_bound) - 1;
    }
    else if(expected_loc[0] < -x_bound) {
        expected_loc[0] = x_bound + (expected_loc[0] + x_bound) + 1;
    }
    else if(expected_loc[1] > y_bound) {
        expected_loc[1] = -y_bound + (expected_loc[1] - y_bound) - 1;
    }
    else if(expected_loc[1] < -y_bound) {
        expected_loc[1] = y_bound + (expected_loc[1] + y_bound) + 1;
    }

    test_ant.move(movement);
    
    auto final_loc = test_ant.get_location(); 

    ASSERT_EQ(expected_loc, final_loc);
}

TEST(grid, ant_on_patch) {
    
    std::default_random_engine engine{static_cast<unsigned int>(testing::UnitTest::GetInstance()->random_seed())};

    std::uniform_int_distribution<int> dist(10, 100);

    std::vector<grid::state> grid_states = {
        {{1.0, 1.0, 1.0}, {-1, 0}},
        {{1.0, 1.0, 1.0}, {1, 0}},
    };

    int x_bound = dist(engine);
    int y_bound = dist(engine);

    grid::grid test_grid(x_bound, y_bound, grid_states);

    Eigen::Vector2i location(0, 0);

    for(auto &column : test_grid.patches) {
        for(auto &this_patch : column) {
            ASSERT_EQ(this_patch.state_index, 0);
        }
    }

    Eigen::Vector2i movement = test_grid.ant_on_patch(location);

    ASSERT_EQ(test_grid.patches[x_bound][y_bound].state_index % grid_states.size(), 1);
    ASSERT_EQ(movement, grid_states[0].ant_relative_direction); 

    movement = test_grid.ant_on_patch(location);
    
    ASSERT_EQ(test_grid.patches[x_bound][y_bound].state_index % grid_states.size(), 0);
    ASSERT_EQ(movement, grid_states[1].ant_relative_direction); 
}
int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
