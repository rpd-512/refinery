#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // for std::vector, std::map bindings
#include <pybind11/functional.h> // for std::function bindings if needed

#include "types.h"
#include "nearest_neighbour_utils.h"
#include "refinement_utils.h"
#include "refiner_content.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "py-refinery PyBind11 module - adaptive inverse problem solving engine";

    // Bind classes
    py::class_<Datapoint>(m, "Datapoint")
        .def(py::init<>())  // default constructor
        .def(py::init<int, const vector<double>&, const vector<double>&>(), 
             py::arg("id"), py::arg("features"), py::arg("groundtruth") = std::vector<double>{})
        .def_readwrite("id", &Datapoint::id)
        .def_readwrite("features", &Datapoint::features)
        .def_readwrite("groundtruth", &Datapoint::groundtruth)
        .def_static("from_vector", &Datapoint::from_vector, py::arg("features"), py::arg("groundtruth") = std::vector<double>{}, py::arg("id") = 0)
        .def_static("to_vector", &Datapoint::to_vector);

    py::class_<NearestNeighbourEngine>(m, "NearestNeighbourEngine")
        .def(py::init<int, std::vector<Datapoint>>(), py::arg("dimensions"), py::arg("datapoints"))
        
        .def("insert", (void (NearestNeighbourEngine::*)(const Datapoint&, bool)) &NearestNeighbourEngine::insert,
             py::arg("datapoint"), py::arg("rebalance") = true)

        .def("insert_batch", (void (NearestNeighbourEngine::*)(const std::vector<Datapoint>&)) &NearestNeighbourEngine::insert,
             py::arg("datapoints"))
        .def("query", &NearestNeighbourEngine::query);
    
    py::class_<Optimizer>(m, "Optimizer")
        .def("optimize", &Optimizer::optimize);  // expose any public virtual functions

    py::class_<GradientDescentOptimizer, Optimizer>(m, "GradientDescentOptimizer")
        .def(py::init<
            std::function<Feature(const Groundtruth&)>,
            std::function<double(const Groundtruth&, const Feature&)>,
            double>(),
            py::arg("forward_function"),
            py::arg("loss_function"),
            py::arg("learning_rate") = 0.01
        )
        .def("optimize", &GradientDescentOptimizer::optimize);
    py::class_<AdamOptimizer, Optimizer>(m, "AdamOptimizer")
        .def(py::init<
            std::function<Feature(const Groundtruth&)>,
            std::function<double(const Groundtruth&, const Feature&)>,
            double,
            double,
            double,
            double>(),
            py::arg("forward_function"),
            py::arg("loss_function"),
            py::arg("learning_rate") = 0.001,
            py::arg("beta1") = 0.9,
            py::arg("beta2") = 0.999,
            py::arg("epsilon") = 1e-5
        )
        .def("optimize", &AdamOptimizer::optimize);

    py::class_<GradientMomentumOptimizer, Optimizer>(m, "GradientMomentumOptimizer")
        .def(py::init<
            std::function<Feature(const Groundtruth&)>,
            std::function<double(const Groundtruth&, const Feature&)>,
            double,
            double>(),
            py::arg("forward_function"),
            py::arg("loss_function"),
            py::arg("learning_rate") = 0.01,
            py::arg("momentum") = 0.9
        )
        .def("optimize", &GradientMomentumOptimizer::optimize);

    py::class_<GradientNesterovOptimizer, Optimizer>(m, "GradientNesterovOptimizer")
        .def(py::init<
            std::function<Feature(const Groundtruth&)>,
            std::function<double(const Groundtruth&, const Feature&)>,
            double,
            double>(),
            py::arg("forward_function"),
            py::arg("loss_function"),
            py::arg("learning_rate") = 0.01,
            py::arg("momentum") = 0.9
        )
        .def("optimize", &GradientNesterovOptimizer::optimize);
    
    py::class_<AdagradOptimizer, Optimizer>(m, "AdagradOptimizer")
        .def(py::init<
            std::function<Feature(const Groundtruth&)>,
            std::function<double(const Groundtruth&, const Feature&)>,
            double,
            double>(),
            py::arg("forward_function"),
            py::arg("loss_function"),
            py::arg("learning_rate") = 0.01,
            py::arg("epsilon") = 1e-8
        )
        .def("optimize", &AdagradOptimizer::optimize);

    py::class_<RMSpropOptimizer, Optimizer>(m, "RMSpropOptimizer")
        .def(py::init<
            std::function<Feature(const Groundtruth&)>,
            std::function<double(const Groundtruth&, const Feature&)>,
            double,
            double,
            double>(),
            py::arg("forward_function"),
            py::arg("loss_function"),
            py::arg("learning_rate") = 0.001,
            py::arg("decay_rate") = 0.99,
            py::arg("epsilon") = 1e-8
        )
        .def("optimize", &RMSpropOptimizer::optimize);

    py::class_<RefinementEngine>(m, "RefinementEngine")
        .def(py::init<Optimizer*>(), py::arg("optimizer"))
        .def("set_seed", &RefinementEngine::set_seed, py::arg("seed_vector"))
        .def("set_target", &RefinementEngine::set_target, py::arg("target_feature"))
        .def("set_logging", &RefinementEngine::set_logging, py::arg("log_flag"))
        .def("refine", &RefinementEngine::refine, py::arg("iterations"))
        .def("get_loss_history", &RefinementEngine::get_loss_history)
        .def("get_data_history", &RefinementEngine::get_data_history);

    py::class_<LossFunction>(m, "LossFunction")
        .def_static("mse_loss", &LossFunction::mse_loss)
        .def_static("mae_loss", &LossFunction::mae_loss)
        .def_static("huber_loss", &LossFunction::huber_loss)
        .def_static("log_cosh_error", &LossFunction::log_cosh_error)
        .def_static("cosine_error", &LossFunction::cosine_error)
        .def_static("euclidean_loss", &LossFunction::euclidean_loss);
}