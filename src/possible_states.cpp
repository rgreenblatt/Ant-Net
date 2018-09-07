#include "possible_states.h"

void generate_all_states(std::vector<std::vector<int>> &possible_states, int min, int max, unsigned int num_states) {

    std::vector<int> values_in_range;

    for(int i = min; i <= max; i++) {
        values_in_range.push_back(i);
    } 

    for(int i = -min; i >= -max; i--) {
        values_in_range.push_back(i);
    } 
    for(auto val_in_range : values_in_range) {
        std::vector<int> initial;
        initial.push_back(val_in_range);
        possible_states.push_back(initial);
    }

    for(int i = 1; i < num_states; i++) {
        for(int k = possible_states.size() - 1; k >=0; k--) {
            bool first = true;
            std::vector<int> new_state_set;
            for(auto val_in_range : values_in_range) {
                if(first) {
                    possible_states[k].push_back(val_in_range);

                    new_state_set = possible_states[k];

                    first = false;
                } else {
                    new_state_set.back() = val_in_range;
                    possible_states.push_back(new_state_set);
                }
            }
        }
    }
}
