#ifndef LEARNER_H
#define LEARNER_H

#include "network.h"
#include "args.h"
#include "math.h"

class Learner{
public:
	double alpha;
	double eta[2];
	int K;
	int N;
	double* beta;
	double** pi;
	double link_ratio;
	double epsilon;
	int step_count;
	vector<double> ppx_held_out;
	int max_iteration;
	Network network;

	Learner(Args arg, Network &network){
		this->network = network;
		this->epsilon = arg.epsilon;
		this->N = network.N;
		this->alpha = arg.alpha;
		this->eta[0] = arg.eta0;
		this->eta[1] = arg.eta1;
		this->K = arg.K;
		this->step_count = 0;
		this->link_ratio = network.total_num_linked_edges/(N*(N-1)/2);
		this->max_iteration = arg.max_iteration;

		beta = new double[K];
		pi = new double*[N];
		for (int i = 0; i < N; i++){
			pi[i] = new double[K];
		}
	}


	double cal_perplexity(set<Edge> &edges){
		double link_likelihood = 0;
		double non_link_likelihood = 0;
		double link_count = 0;
		double non_link_count = 0;
		double avg_likelihood = 0;
		for (std::set<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it){
			  Edge e = *it;
			  std::set<Edge>::iterator it1;
			  int y = 0;
			  it1 = this->network.link_edges.find(e);
			  if (it1 != this->network.link_edges.end()){
				  y = 1;
			  }

			  double edge_likelihood = cal_edge_likelihood(e.u,e.v,y);
			  if (y==1){
				  link_count += 1;
				  link_likelihood += edge_likelihood;
			  }else{
				  non_link_count += 1;
				  non_link_likelihood += edge_likelihood;
			  }
		}

		//avg_likelihood = link_ratio*(link_likelihood/link_count) +
		 //                          (1-link_ratio)*(non_link_likelihood/non_link_count);
		avg_likelihood = (link_likelihood + non_link_likelihood)/(link_count+non_link_count);
		cout<<"perplexity is: "<<avg_likelihood<<endl;
		return -avg_likelihood;

	}

	double cal_edge_likelihood(int a, int b, int y){
		double s = 0;
		if (y==1){
			for (int k = 0; k < K; k++){
				s += pi[a][k] * pi[b][k] * beta[k];
			}
		}else{
			double sum = 0;
			for (int k = 0; k < K; k++){
				 s += pi[a][k] * pi[b][k] * (1-beta[k]);
				 sum += pi[a][k]*pi[b][k];
			}
			s+= (1-sum)*(1-epsilon);
		}

		if (s < 1e-30){
			s = 1e-30;
		}

		return log(s);
	}
};

#endif

