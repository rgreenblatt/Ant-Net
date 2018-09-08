#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <random>

inline Eigen::Vector2d rotate_d(Eigen::Vector2d to_be_rotated, Eigen::Vector2i rotater) {
    Eigen::Vector2d retval(to_be_rotated[0] * rotater[0] - to_be_rotated[1] * rotater[1], to_be_rotated[0] * rotater[1] + to_be_rotated[1] * rotater[0]);
    return retval;
}

inline Eigen::Vector2i rotate_i(Eigen::Vector2i to_be_rotated, Eigen::Vector2i rotater) {
    Eigen::Vector2i retval(to_be_rotated[0] * rotater[0] - to_be_rotated[1] * rotater[1], to_be_rotated[0] * rotater[1] + to_be_rotated[1] * rotater[0]);
    return retval;
}

namespace ant {
    class ant_interface {
    public:
        ant_interface(int x_bound_in, int y_bound_in, bool randomize = false) {
            if(randomize) {
                std::random_device rd;
                std::default_random_engine engine{rd()};

                std::uniform_int_distribution<int> x_dist(-x_bound_in, x_bound_in);
                std::uniform_int_distribution<int> y_dist(-y_bound_in, y_bound_in);

                std::uniform_int_distribution<int> x_or_y_dist(0, 1);
                std::uniform_int_distribution<int> sign_dist(0, 1);

                location[0] = x_dist(engine);
                location[1] = y_dist(engine);
                
                int x_or_y = x_or_y_dist(engine);
                orientation[x_or_y] = sign_dist(engine) ? -1 : 1;
                orientation[!x_or_y] = 0;
            } else {
                location[0] = 0;
                location[1] = 0;

                orientation[0] = 0;
                orientation[1] = 1;
            }

            x_bound = x_bound_in;
            y_bound = y_bound_in;
        };
        virtual ~ant_interface() {};
        
        virtual void move(Eigen::Vector2i ant_relative_direction) = 0;  
        
        int get_state() { 
            return state;
        }

        Eigen::Vector2i get_location() { 
            return location;
        }

        Eigen::Vector2i get_orientation() { 
            return orientation;
        }

        static std::vector<Eigen::Vector3d> ant_color_state;
    protected: 
        int state = 0;
        Eigen::Vector2i location; 
        Eigen::Vector2i orientation; 
        int x_bound = 0;
        int y_bound = 0;
    };

    class base_ant : public ant_interface { 
    public:
        base_ant(int x_bound_in, int y_bound_in, bool randomize = false) : ant_interface(x_bound_in, y_bound_in, randomize) {}
        void move(Eigen::Vector2i ant_relative_direction);
    };
}
