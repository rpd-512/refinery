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
    
    Datapoint dp;
    dp.id = 1;
    dp.features = {10.5, 20.3, 30.7};
    dp.groundtruth = {11.0, 21.0, 31.0};

    NearestNeighbourEngine engine(3, {});
    int n=1000;
    vector<Datapoint> initial_points;
    for(int i=0; i<n; i++){
        initial_points.push_back(Datapoint{
            i+2,
            {uniform(0, 20), uniform(0, 40), uniform(0, 50)},
            {uniform(-10, 10), uniform(-10, 10), uniform(-10, 10)}
        });
    }
    engine.insert(initial_points);
    cout << "Inserted " << n << " datapoints." << endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    cout << "Query Datapoint:" << endl;
    print_datapoint(dp);

    //measure time
    start = std::chrono::high_resolution_clock::now();
    cout << "KD tree search result:" << endl;
    Datapoint result_qry = engine.query(dp);
    print_datapoint(result_qry);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "KD tree search time: " << duration.count() << " microseconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    cout << "Linear search result:" << endl;
    Datapoint result_lin = engine.linear_search(dp);
    print_datapoint(result_lin);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Linear search time: " << duration.count() << " microseconds" << std::endl;

    cout << "\n\nTesting Refinement Engine with Gradient Descent Optimizer\n";


    auto fwd_func = [](const Groundtruth& gt) {
        cout << "Forward function called with groundtruth of size: " << gt.size() << endl;
        Feature f = {1,2,3};
        return f;
    };
    Feature target = {2,3,5};
    GradientDescentOptimizer optimizer(fwd_func, LossFunction::cosine_error);
    RefinementEngine ref_engine(&optimizer);
    ref_engine.set_seed(dp);
    ref_engine.set_logging(true);
    ref_engine.set_target(target);
    auto refined = ref_engine.refine(100);
    cout << "Refined Datapoint:" << endl;
    print_datapoint(Datapoint::from_vector(refined));
    for(auto his: ref_engine.get_loss_history()) {
        cout << his << ", ";
    }
    cout << endl;
    return 0;
}