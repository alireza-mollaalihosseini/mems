#include </home/almo2783/work/eigen-3.4.0/eigen-3.4.0/Eigen/Dense>
#include </home/almo2783/work/fftw-include/include/fftw3.h>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <tuple>
#include <numeric>


// Function to calculate derivatives
std::vector<double> derivatives(double t, std::vector<double>& states, double a, double k, double omega_0,
                                 double Q_0, double alpha, double beta, double gamma, double R,
                                 double tau, double u_dc, double u_max) {
    double x     = states[0];
    double x_dot = states[1];
    double theta = states[2];
    double u_ac  = states[3];

    double u_act = a * u_ac + u_dc;

    double dx_dt     = x_dot;
    double dx_dot_dt = -(omega_0 / Q_0) * x_dot - (omega_0 * omega_0) * x + alpha * theta;
    double dtheta_dt = -(beta * theta) + (gamma * std::min(u_act * u_act, u_max * u_max)) / (R * R);
    double du_ac_dt  = -(u_ac / tau) + (k * x_dot);

    std::vector<double> result = {dx_dt, dx_dot_dt, dtheta_dt, du_ac_dt};
    return result;
}

// Function to calculate derivatives
std::vector<double> derivatives_ext(double t, std::vector<double>& states, double a, double k, double omega_0,
                                 double Q_0, double alpha, double beta, double gamma, double R,
                                 double tau, double u_dc, double u_max, double F_ext, double mu) {
    double x     = states[0];
    double x_dot = states[1];
    double theta = states[2];
    double u_ac  = states[3];

    double u_act = a * u_ac + u_dc;

    double dx_dt     = x_dot;
    double dx_dot_dt = -(omega_0 / Q_0) * x_dot - (omega_0 * omega_0) * x + alpha * theta + mu * F_ext;
    double dtheta_dt = -(beta * theta) + (gamma * std::min(u_act * u_act, u_max * u_max)) / (R * R);
    double du_ac_dt  = -(u_ac / tau) + (k * x_dot);

    std::vector<double> result = {dx_dt, dx_dot_dt, dtheta_dt, du_ac_dt};
    return result;
}

// Fourth-order Runge-Kutta method
std::vector<std::vector<double>> rungeKutta(double t0, std::vector<double>& y0, double t, double h,
                                             double a, double k, double omega_0, double Q_0,
                                             double alpha, double beta, double gamma, double R,
                                             double tau, double u_dc, double u_max, 
                                             const std::vector<double>& f_ext, double t_f, double mu) {
    // Count the number of iterations
    int n = static_cast<int>((t - t0) / h);
    int n1 = static_cast<int>((t_f - t0) / h);

    std::vector<std::vector<double>> y_values;
    y_values.push_back(y0);

    std::vector<double> y = y0;

    std::vector<double> k1, k2, k3, k4;
    std::vector<double> temp;

    // Iterate for n steps
    for (int i = 0; i < n1; i++) {
        k1 = derivatives(t0, y, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max);
        temp = y;
        for (size_t j = 0; j < y.size(); j++) {
            temp[j] = y[j] + 0.5 * h * k1[j];
        }

        k2 = derivatives(t0 + 0.5 * h, temp, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max);
        for (size_t j = 0; j < y.size(); j++) {
            temp[j] = y[j] + 0.5 * h * k2[j];
        }

        k3 = derivatives(t0 + 0.5 * h, temp, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max);
        for (size_t j = 0; j < y.size(); j++) {
            temp[j] = y[j] + h * k3[j];
        }

        k4 = derivatives(t0 + h, temp, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max);

        // Update next value of y
        for (size_t j = 0; j < y.size(); j++) {
            y[j] = y[j] + (1.0/6.0) * h * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
        }

        // Update next value of t
        t0 = t0 + h;

        // Save y to y_values
        y_values.push_back(y);
    }

    // Iterate for n steps
    for (int i = n1; i < n; i++) {
        k1 = derivatives_ext(t0, y, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max, f_ext[i-n1], mu);
        temp = y;
        for (size_t j = 0; j < y.size(); j++) {
            temp[j] = y[j] + 0.5 * h * k1[j];
        }

        k2 = derivatives_ext(t0 + 0.5 * h, temp, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max, f_ext[i-n1], mu);
        for (size_t j = 0; j < y.size(); j++) {
            temp[j] = y[j] + 0.5 * h * k2[j];
        }

        k3 = derivatives_ext(t0 + 0.5 * h, temp, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max, f_ext[i-n1], mu);
        for (size_t j = 0; j < y.size(); j++) {
            temp[j] = y[j] + h * k3[j];
        }

        k4 = derivatives_ext(t0 + h, temp, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max, f_ext[i-n1], mu);

        // Update next value of y
        for (size_t j = 0; j < y.size(); j++) {
            y[j] = y[j] + (1.0/6.0) * h * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]);
        }

        // Update next value of t
        t0 = t0 + h;

        // Save y to y_values
        y_values.push_back(y);
    }

    return y_values;
}

// Function to get the last column
std::vector<double> get_last_column(const std::vector<std::vector<double>>& result) {
    std::vector<double> last_column;
    
    // Iterate over each row and extract the last element
    for (const auto& row : result) {
        if (!row.empty()) {
            last_column.push_back(row.back()); // Access the last element of each row
        }
    }
    
    return last_column;
}

std::vector<double> interpolate(const std::vector<double>& vector, double fraction) {
    int original_length = vector.size();
    int new_length = static_cast<int>(original_length * fraction);
    std::vector<double> interpolated_vector(new_length, 0.0);

    for (int i = 0; i < original_length - 1; ++i) {
        int start_index = static_cast<int>(i * fraction);
        int end_index = static_cast<int>((i + 1) * fraction);
        interpolated_vector[start_index] = vector[i];
        for (int j = start_index + 1; j < end_index; ++j) {
            interpolated_vector[j] = vector[i];
        }
    }

    // Copying the last value
    interpolated_vector[new_length - 1] = vector[original_length - 1];

    return interpolated_vector;
}

// Function to generate the FIR lowpass filter coefficients
std::vector<double> fir_lowpass(double cutoff, double fs, int order = 101) {

    double nyquist = 0.5 * fs;
    double cutoff_normalized = cutoff / nyquist;
    double omega_c = 2.0 * M_PI * cutoff_normalized;
    int N = order;

    // FIR filter coefficients calculation using window method
    std::vector<double> b(N);
    double sum = 0.0;
    for (int n = 0; n < N; ++n) {
        if (n == N / 2) {
            b[n] = 2.0 * cutoff_normalized;
        } else {
            double idx = n - N / 2;
            b[n] = sin(omega_c * idx) / (M_PI * idx);
        }
        sum += b[n];
    }

    // Normalize the coefficients
    for (int n = 0; n < N; ++n) {
        b[n] /= sum;
    }

    return b;
}

// Function to apply the lowpass filter to the data
std::vector<double> apply_lowpass_filter(const std::vector<double>& data, double cutoff, double fs, int order = 101) {

    std::vector<double> b = fir_lowpass(cutoff, fs, order);
    std::vector<double> y(data.size(), 0.0);

    for (int n = 0; n < data.size(); ++n) {
        for (int k = 0; k < b.size(); ++k) {
            if (n - k >= 0) {
                y[n] += b[k] * data[n - k];
            }
        }
    }

    return y;
}

// Define a struct to hold WAV file header information
struct WavHeader {
    char chunkId[4];
    uint32_t chunkSize;
    char format[4];
    char subchunk1Id[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2Id[4];
    uint32_t subchunk2Size;
};

bool loadWavFile(const std::string& filename, std::vector<double>& audioData, uint32_t& sampleRate) {
    // Open the WAV file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening WAV file." << std::endl;
        return false;
    }

    // Read the WAV header
    WavHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        std::cerr << "Error reading WAV header." << std::endl;
        return false;
    }

    // Check the file format and configuration
    if (std::string(header.chunkId, 4) != "RIFF" || std::string(header.format, 4) != "WAVE" ||
        header.audioFormat != 1 || header.numChannels != 1 || header.bitsPerSample != 24 ||
        header.sampleRate != 44100) {
        std::cerr << "Unsupported WAV file format or configuration." << std::endl;
        return false;
    }

    // Calculate the number of samples
    uint32_t numSamples = header.subchunk2Size / (header.bitsPerSample / 8);

    // Resize the output audio data vector
    audioData.resize(numSamples);

    // Read audio samples
    for (size_t i = 0; i < numSamples; ++i) {
        uint8_t sampleBytes[3] = {0}; // 24-bit sample (3 bytes)
        file.read(reinterpret_cast<char*>(sampleBytes), 3);
        if (!file) {
            std::cerr << "Error reading audio sample." << std::endl;
            return false;
        }

        // Convert 24-bit sample to 32-bit signed integer
        int32_t sampleValue = (sampleBytes[0] | (sampleBytes[1] << 8) | (sampleBytes[2] << 16));
        if (sampleBytes[2] & 0x80) { // Check the sign bit
            sampleValue |= 0xFF000000; // Sign extend to 32 bits
        }

        // Store the sample in the audio data vector
        audioData[i] = static_cast<double>(sampleValue);
    }

    // Close the file
    file.close();

    // Set the sample rate
    sampleRate = header.sampleRate;

    // Rescale audio data to the range [-1, 1]
    double maxAmplitude = *std::max_element(audioData.begin(), audioData.end(), 
                                            [](double a, double b) { return std::abs(a) < std::abs(b); });
    if (maxAmplitude != 0) {  // Prevent division by zero
        for (auto& sample : audioData) {
            sample /= maxAmplitude;
        }
    }

    return true;
}

// Function to calculate the mean of a vector of numbers
double calculateMean(const std::vector<double>& numbers) {
    // Check if the vector is empty to avoid division by zero
    if (numbers.empty()) {
        throw std::invalid_argument("The input vector is empty");
    }

    // Sum all the elements in the vector
    double sum = std::accumulate(numbers.begin(), numbers.end(), 0.0);

    // Calculate the mean
    double mean = sum / numbers.size();

    return mean;
}

// Function to normalize data to the range [-1, 1]
void normalize_data(std::vector<double>& data) {
    // Find the maximum absolute value in data
    double max_abs = 0.0;
    for (const auto& val : data) {
        double abs_val = std::abs(val);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    // Normalize data
    if (max_abs > 0.0) {
        for (auto& val : data) {
            val /= max_abs;
        }
    }
}

// Function to calculate the maximum absolute value in a dataset
double max_abs_value(const std::vector<double>& dataset) {
    double max_abs = 0.0;
    for (const auto& val : dataset) {
        double abs_val = std::abs(val);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    return max_abs;
}

// Function to scale a dataset by a given scaling factor
std::vector<double> scale_dataset(const std::vector<double>& dataset, double scaling_factor) {
    std::vector<double> scaled_dataset;
    scaled_dataset.reserve(dataset.size());
    for (const auto& val : dataset) {
        scaled_dataset.push_back(val * scaling_factor);
    }
    return scaled_dataset;
}

// Function to convert Eigen::MatrixXd to std::vector<std::vector<double>>
std::vector<std::vector<double>> eigen_to_vector(const Eigen::MatrixXd& matrix) {
    std::vector<std::vector<double>> vec(matrix.rows(), std::vector<double>(matrix.cols()));
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            vec[i][j] = matrix(i, j);
        }
    }
    return vec;
}

// Function to save matrix to file
void save_matrix_to_file(const std::string& filename, const std::vector<std::vector<double>>& matrix) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for writing!" << std::endl;
        return;
    }
    for (const auto& row : matrix) {
        for (double val : row) {
            file << val << " ";
        }
        file << "\n";
    }
    file.close();
}

// Function to load matrix from file
std::vector<std::vector<double>> load_matrix_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filename << " for reading!" << std::endl;
        return {};
    }
    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::istringstream iss(line);
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

// Function to calculate frequency (f), angular frequency (omega_0), Reynolds number (Re), and quality factor (Q_0)
void calculate_parameters(double d_si, double l_si, double w_si, double delta_n, double E_si, double rho_si, double rho_gas, double eta_gas, double& f, double& omega_0) {
    // Calculate f
    f = std::pow(delta_n, 2) * (d_si / (2 * M_PI * std::pow(l_si, 2))) * std::sqrt(E_si / (12 * rho_si));

    // Calculate omega_0
    omega_0 = 2 * M_PI * f;
}

void processDataset(const std::string &inputFilename, const std::string &stateMatrixPath, const std::string &labelMatrixPath, 
                    double a, double u_dc, double h, int Nodes, double t0, std::vector<double>& y0, double t_end, double k, double omega_0, double Q_0,
                    double alpha, double beta, double gamma, double R, double tau, double u_max, double t_ext, double mu) {

    std::ifstream filenamesFile(inputFilename);
    if (!filenamesFile.is_open()) {
        std::cerr << "Error opening " << inputFilename << " file!" << std::endl;
        return;
    }

    std::ofstream stateMatrixFile(stateMatrixPath);
    std::ofstream labelMatrixFile(labelMatrixPath);

    if (!stateMatrixFile.is_open() || !labelMatrixFile.is_open()) {
        std::cerr << "Error opening output file for writing!" << std::endl;
        return;
    }

    std::string filename;
    while (std::getline(filenamesFile, filename)) {
        std::ifstream file(filename);
        if (file.good()) {
            std::vector<std::string> parts;
            std::istringstream iss(filename);
            std::string token;
            while (std::getline(iss, token, '/')) {
                parts.push_back(token);
            }

            std::string lastSegment = parts.back();
            std::istringstream iss2(lastSegment);
            std::vector<std::string> info;
            while (std::getline(iss2, token, '-')) {
                info.push_back(token);
            }

            std::string scene = info[0];
            std::vector<double> audioData;
            uint32_t sampleRate;

            if (loadWavFile(filename, audioData, sampleRate)) {

                // Interpolate the dataset
                double new_sampleRate = 1000000;
                double fraction = new_sampleRate / sampleRate;
                std::vector<double> interpolated_vector = interpolate(audioData, fraction);

                // Apply lowpass filter
                double cutoff = sampleRate / 2;
                double fs = new_sampleRate;
                int order = 101;
                std::vector<double> filtered_data = apply_lowpass_filter(interpolated_vector, cutoff, fs, order);

                // Solve using Runge-Kutta method
                std::vector<std::vector<double>> result = rungeKutta(t0, y0, t_end, h, a, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_dc, u_max, filtered_data, t_ext, mu);
                std::vector<double> voltages = get_last_column(result);
                std::vector<double> cropped_voltages(voltages.begin() + static_cast<size_t>(2.5 / h), voltages.begin() + static_cast<size_t>(3.5 / h));
                double voltages_mean = calculateMean(cropped_voltages);

                std::vector<double> signal(cropped_voltages.size());
                for (size_t k = 0; k < cropped_voltages.size(); ++k) {
                    signal[k] = cropped_voltages[k] - voltages_mean;
                }

                int N = signal.size();
                fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
                fftw_plan plan = fftw_plan_dft_r2c_1d(N, signal.data(), out, FFTW_ESTIMATE);
                fftw_execute(plan);

                for (int k = 0; k < Nodes; ++k) {
                    double magnitude = std::sqrt(out[k][0] * out[k][0] + out[k][1] * out[k][1]);
                    stateMatrixFile << magnitude << " ";
                }

                stateMatrixFile << 1.0 << std::endl;

                if (scene == "airport") {
                    labelMatrixFile << "1 0 0 0 0 0 0 0 0 0" << std::endl;
                } else if (scene == "shopping_mall") {
                    labelMatrixFile << "0 1 0 0 0 0 0 0 0 0" << std::endl;
                } else if (scene == "metro_station") {
                    labelMatrixFile << "0 0 1 0 0 0 0 0 0 0" << std::endl;
                } else if (scene == "street_pedestrian") {
                    labelMatrixFile << "0 0 0 1 0 0 0 0 0 0" << std::endl;
                } else if (scene == "public_square") {
                    labelMatrixFile << "0 0 0 0 1 0 0 0 0 0" << std::endl;
                } else if (scene == "street_traffic") {
                    labelMatrixFile << "0 0 0 0 0 1 0 0 0 0" << std::endl;
                } else if (scene == "tram") {
                    labelMatrixFile << "0 0 0 0 0 0 1 0 0 0" << std::endl;
                } else if (scene == "bus") {
                    labelMatrixFile << "0 0 0 0 0 0 0 1 0 0" << std::endl;
                } else if (scene == "metro") {
                    labelMatrixFile << "0 0 0 0 0 0 0 0 1 0" << std::endl;
                } else if (scene == "park") {
                    labelMatrixFile << "0 0 0 0 0 0 0 0 0 1" << std::endl;
                } else {
                    labelMatrixFile << "0 0 0 0 0 0 0 0 0 0" << std::endl;
                }

                fftw_destroy_plan(plan);
                fftw_free(out);
            }
        }
    }

    filenamesFile.close();
    stateMatrixFile.close();
    labelMatrixFile.close();
}


// Main function
int main(int argc, char *argv[]) {

    // Parameters
    double t0 = 0.0;
    double t_end = 3.5;
    double h = 0.000001;
    double t_ext = 2.5;
    std::vector<double> y0 = {std::pow(10, -9), 0.0, 0.0, 0.0};

    // Additional parameters
    double w_si = 150e-6;
    double l_si = 350e-6;
    double d_si = 1.25e-6;
    double rho_si = 2329;
    double E_si = 170e-6;
    double delta_1 = 1.8751;
    double rho_gas = 1.189;
    double eta_gas = 18.232e-6;

    double f = 14e3;
    double Q_0 = 43.2;
    double alpha = 749.37;
    double beta = 1006.6;
    double gamma = 4.2588e7;
    double R = 25.0;
    double tau = 0.001;
    double k = 1e6;
    double u_max = 1.0;
    double omega_0 = 87964.6;

    // Parse the command-line arguments
    // double a = std::atof(argv[1]);     // 'a' value
    // double u_dc = std::atof(argv[2]);  // 'u_dc' value
    // double mu = std::atof(argv[3]);    // 'mu' value (fixed at 1.0)
    // double lambda = 1e4;
    // int Nodes = 24001;

    double a      = -1.08;
    double u_dc   = 0.1;
    double mu     = 1.7e-5;
    double lambda = 1e4;
    int Nodes     = 24001;

    std::stringstream ss, su;
    ss << std::fixed << std::setprecision(2) << a;
    su << std::fixed << std::setprecision(1) << u_dc;

    // Construct file paths
    std::string X_train = "/scratch/almo2783/scratch/rayson/design1/3cities/state-matrices/X_train_a_" + ss.str() + "_u_dc_" + su.str() + ".txt";
    std::string y_train = "/scratch/almo2783/scratch/rayson/design1/3cities/state-matrices/y_train_a_" + ss.str() + "_u_dc_" + su.str() + ".txt";
    std::string X_val   = "/scratch/almo2783/scratch/rayson/design1/3cities/state-matrices/X_val_a_" + ss.str() + "_u_dc_" + su.str() + ".txt";
    std::string y_val   = "/scratch/almo2783/scratch/rayson/design1/3cities/state-matrices/y_val_a_" + ss.str() + "_u_dc_" + su.str() + ".txt";
    std::string X_test  = "/scratch/almo2783/scratch/rayson/design1/3cities/state-matrices/X_test_a_" + ss.str() + "_u_dc_" + su.str() + ".txt";
    std::string y_test  = "/scratch/almo2783/scratch/rayson/design1/3cities/state-matrices/y_test_a_" + ss.str() + "_u_dc_" + su.str() + ".txt";

    // Process the dataset
    processDataset("/scratch/almo2783/scratch/rayson/design1/3cities/train-filenames-3cities-rayson.csv", X_train, y_train,
                   a, u_dc, h, Nodes, t0, y0, t_end, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_max, t_ext, mu);

    processDataset("//scratch/almo2783/scratch/rayson/design1/3cities/val-filenames-3cities-rayson.csv", X_val, y_val,
                   a, u_dc, h, Nodes, t0, y0, t_end, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_max, t_ext, mu);

    processDataset("/scratch/almo2783/scratch/rayson/design1/3cities/test-filenames-barcelona-rayson.csv", X_test, y_test,
                   a, u_dc, h, Nodes, t0, y0, t_end, k, omega_0, Q_0, alpha, beta, gamma, R, tau, u_max, t_ext, mu);

    // Call the Python script
    std::string command = "python3 ridge_regression.py " + 
                          std::to_string(a) + " " + 
                          std::to_string(u_dc) + " " + 
                          std::to_string(lambda);
    int result = system(command.c_str());

    if (result != 0) {
        std::cerr << "Error: Python script execution failed with code " << result << std::endl;
    }

    // // Delete files after calculations
    // if (std::remove(X_train.c_str()) != 0) {
    //     std::cerr << "Error deleting X_train file: " << std::strerror(errno) << std::endl;
    // }
    // if (std::remove(y_train.c_str()) != 0) {
    //     std::cerr << "Error deleting y_train file: " << std::strerror(errno) << std::endl;
    // }
    // if (std::remove(X_test.c_str()) != 0) {
    //     std::cerr << "Error deleting X_test file: " << std::strerror(errno) << std::endl;
    // }
    // if (std::remove(y_test.c_str()) != 0) {
    //     std::cerr << "Error deleting y_test file: " << std::strerror(errno) << std::endl;
    // }

    return 0;
}