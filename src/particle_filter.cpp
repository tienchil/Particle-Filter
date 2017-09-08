/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	default_random_engine gen;
	num_particles = 10;
	
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 1; i <= num_particles; i++) {
		Particle particle = {};
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		weights.push_back(1.0);
	}

	cout << "Number of Particles: " << num_particles << endl;
	cout << "Weights size: " << weights.size() << endl;

	is_initialized = true;

	cout << "Particle Filter Initialized..." << endl;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	// cout << "Prediction..." << endl;

	for (int i = 0; i < num_particles; i++) {
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);

		double theta_change = particles[i].theta + yaw_rate * delta_t;

		if (fabs(yaw_rate) > 1e-6) {
			// yaw_rate not too small
			double x_f = (velocity/yaw_rate) * (sin(theta_change) - sin(particles[i].theta));
			double y_f = (velocity/yaw_rate) * (cos(particles[i].theta) - cos(theta_change));

			particles[i].x += x_f;
			particles[i].y += y_f;
			particles[i].theta += yaw_rate * delta_t;
		} else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}
	}
	// cout << "Done!..." << endl;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// cout << "Updating Weights..." << endl;
	// Homogenous Transformation
	for(int i = 0; i < num_particles; ++i) {

		double weight = 1.0;

		for (int j = 0; j < observations.size(); ++j) {
			int obs_id = observations[j].id;
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;

			double part_x = particles[i].x;
			double part_y = particles[i].y;
			double theta = particles[i].theta;

			double x_m = part_x + (cos(theta) * obs_x) - (sin(theta) * obs_y);
			double y_m = part_y + (sin(theta) * obs_x) + (cos(theta) * obs_y);

			// Match landmark
			Map::single_landmark_s best_match = {};
			vector<Map::single_landmark_s> lm_list = map_landmarks.landmark_list;
			double best_dist = 50.0;
			for (int k = 0; k < lm_list.size(); ++k) {
				double temp = dist(x_m, y_m, lm_list[k].x_f, lm_list[k].y_f);

				if (temp <= sensor_range) {
					if (temp < best_dist) {
						best_dist = temp;
						best_match.id_i = lm_list[k].id_i;
						best_match.x_f = lm_list[k].x_f;
						best_match.y_f = lm_list[k].y_f;
					}
				}
			}

			// Update the weight
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double mu_x = best_match.x_f;
			double mu_y = best_match.y_f;

			double gauss_norm = 1.0  / (2.0 * M_PI * sig_x * sig_y);
			double exponent_x = ((x_m - mu_x)*(x_m - mu_x)) / (2.0 * sig_x*sig_x);
			double exponent_y = ((y_m - mu_y)*(y_m - mu_y)) / (2.0 * sig_y*sig_y);

			double new_weight = gauss_norm * exp(-(exponent_x+exponent_y));
					 
			weight *= new_weight;

		}
		// cout << weight << endl;
		weights[i] = weight;
		particles[i].weight = weight;
	}
	// cout << "Done!..." << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	// cout << "Resampling..." << endl;

	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> d(weights.begin(), weights.end());
	
	vector<Particle> particles_temp;

	for (int i = 0; i < num_particles; ++i) {
		weights[i] = 1.0;
		particles[i].weight = 1.0;
		particles_temp.push_back(particles[d(gen)]);
	}

	particles = particles_temp;
	// cout << "Done!..." << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
