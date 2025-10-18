#include "include/types.h"
#include "include/debug_utils.h"
#include "include/nearest_neighbour_utils.h"
#include "include/worker_utils.h"
#include "include/refinement_utils.h"
#include "include/refiner_content.h"
#include <chrono>

void clear_screen(){
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

int main(){
    clear_screen();
    cout << "           __ _                       \n";
    cout << " _ __ ___ / _(_)_ __   ___ _ __ _   _ \n";
    cout << "| '__/ _ \\ |_| | '_ \\ / _ \\ '__| | | |\n";
    cout << "| | |  __/  _| | | | |  __/ |  | |_| |\n";
    cout << "|_|  \\___|_| |_|_| |_|\\___|_|   \\__, |\n";
    cout << "                                |___/ \n";
    cout << "A simple nearest refiner library for general usecase\n";

    vector<Datapoint> data = read_csv("/home/rapidfire69/rapid/coding/researchWorks/inverse_kinematics/dataset_generation_and_compilation/kaggle_upload/kuka_youbot.csv");

    vector<double> search_vector = {20, -10, 100};

    NearestNeighbourEngine nne(3, data);
    Datapoint result = nne.query(Datapoint::from_vector(search_vector));
    print_datapoint(result);

    vector<dh_param> robot = loadDHFromYAML("/home/rapidfire69/rapid/coding/researchWorks/inverse_kinematics/metaheuristic_algorithms/FORGE/example_dh_parameters/kuka_youbot.yaml");
    auto fwd_func = [&robot](const Feature& f) -> Feature {
        return forward_kinematics(f, robot);
    };

    GradientDescentOptimizer sgd(
        fwd_func,
        LossFunction::mse_loss,
        0.00001
    );

    RefinementEngine re(&sgd);
    re.set_seed(result);
    re.set_target(search_vector);
    Groundtruth refined = re.refine(2000);
    cout << "\n\nRefined Result: " << endl;
    cout << "Expected Position: \n\t";
    print_vector(search_vector);
    cout << "Refined Angles: \t";
    print_vector(refined);
    cout << "Forward Kinematics of Refined Angles:\n\t";
    print_vector(fwd_func(refined));
    return 0;
}