/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using namespace std;
using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  num_particles = 80; 
  
  // We reserve the size for vector the compiler will optimize space allocation. 
  // Resize also will work and will be faster i guess.Resize failed to execute correctly.
  // Doesnt make sense check it.
  weights.reserve(num_particles);
  particles.reserve(num_particles);
  
  
  std::default_random_engine gen;
 
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  
   // Initializes particles - from the normal distributions set above
  for (int i = 0; i < num_particles; ++i) {
      
    // Add generated particle data to particles class
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
      
  }
    
  // Show as initialized; 
  is_initialized = true;
  

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
 
  std::default_random_engine gen;
  
  // Make distributions for adding noise
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  
  // Different equations based on if yaw_rate is zero or not
  for (int i = 0; i < num_particles; ++i) {
    
    if (abs(yaw_rate) >= 0.00001) {
      // Add measurements to particles
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
      
    } else {
      // Add measurements to particles
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
     
      
    }

    // Add noise to the particles
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
   // O(N^2) Complexity. Min distance algorithm
   for(auto& obs: observations){
    double min = std::numeric_limits<float>::max();

    for(const auto& pred: predicted){
      double distance = dist(obs.x, obs.y, pred.x, pred.y);
      if( min > distance){
        min = distance;
        obs.id = pred.id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

    for(auto& p: particles){
      p.weight = 1.0;

      // collect  landmarks within sensor range
      vector<LandmarkObs> predictions;
      for(const auto& l: map_landmarks.landmark_list){
        double distance = dist(p.x, p.y, l.x_f, l.y_f);
        if( distance < sensor_range){ 
          predictions.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
        }

      }

      // step 2: Homogeneous Transformation observations coordinates from vehicle to map
      vector<LandmarkObs> map_observations;
      double cos_theta = cos(p.theta);
      double sin_theta = sin(p.theta);

      for(const auto& obs: observations){
        LandmarkObs tmp_obs;
        tmp_obs.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
        tmp_obs.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
        tmp_obs.id = obs.id; 
        map_observations.push_back(tmp_obs);
      }

      // find landmark index for each observation
      dataAssociation(predictions, map_observations);

      // compute the particle's weight
      // We keep multiplying weights which can make the weight really small may be we can use log values 
      //of probability for better computation speeds and accuracy
      for(const auto& obs_m: map_observations){

        Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);
        double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
        double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
        double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        p.weight *=  w;
      }

      weights.push_back(p.weight);

  }
}

void ParticleFilter::resample() {
 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(weights.begin(), weights.end());

  // create resampled particles
  //Vector reserve seems to break the code. Fix it.
  vector<Particle> resampled_particles;
  resampled_particles.resize(num_particles);

  // resample the particles according to weights
  for(int i=0; i<num_particles; i++){
    int idx = dist(gen);
    resampled_particles[i] = particles[idx];
  }

  // assign the resampled_particles to the previous particles
  particles = resampled_particles;

  // clear the weight vector for the next round
  weights.clear();

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}