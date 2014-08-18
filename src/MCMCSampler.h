#ifndef MCMC_SAMPLER
#define MCMC_SAMPLER

#include "args.h"
#include "network.h"
#include <algorithm>
#include <cmath>
#include "learner.h"
class MCMCSampler: public Learner{
public:
	double a;
	double b;
	double c;
	int num_node_samples;
	double ** theta;
	double** phi;
	vector<double> avg_likelihood;
	vector<double> timing;

	MCMCSampler(Args arg, Network &network): Learner(arg, network){

		this->a = arg.a;
		this->b = arg.b;
		this->c = arg.c;
		this->num_node_samples = 30;

		theta = new double*[K];
		for (int i = 0; i < K; i++){
			theta[i] = new double[2]();
		}

		gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
		gsl_rng_set(r, time(NULL));

		for (int k = 0; k < K; k++){
			for (int t = 0; t < 2; t++){
				theta[k][t] = gsl_ran_gamma(r,1,1);
			}
		}

		phi = new double*[N];
		for (int i = 0; i < N; i++){
			phi[i] = new double[K]();
		}

		for (int i = 0; i < N; i++){
			for (int k = 0; k < K; k++){
				phi[i][k] = gsl_ran_gamma(r,1,1);
			}
		}

		// update pi and beta
		for (int k = 0; k < K; k++){
			double sum = 0;
			for (int t = 0; t < 2; t++){
				sum += theta[k][t];
			}
			// beta[k] = theta[k][1]/(theta[k][0] + theta[k][1])
			beta[k] = theta[k][1]/sum;
		}

		for (int i = 0; i < N; i++){
			double sum = 0;
			for (int k = 0; k < K; k++){
				sum += phi[i][k];
			}
			for (int k = 0; k < K; k++){
				pi[i][k]= phi[i][k]/sum;
				//cout<<pi[i][k]<< " ";
			}
			//cout<<endl;
		}
	}

	void run(){

		set<Edge> held_out = network.held_out_edges;
		for (std::set<Edge>::iterator it=held_out.begin(); it!=held_out.end(); ++it){
			Edge e = *it;
			//cout<<e.u<<""<<e.v<<endl;
		}

		this->max_iteration = 1000;
		while (this->step_count < this->max_iteration){
			step_count++;
			double eps_t = a*pow(1 + step_count*1.0/b,-c);
			MiniBatch miniBatch = this->network.sample_mini_batch();
			int scale = miniBatch.scale;
			int s = miniBatch.mini_batch_edges.size();

			set<int> nodes = nodes_in_batch(miniBatch.mini_batch_edges);
			for (std::set<int>::iterator it=nodes.begin(); it!=nodes.end(); ++it){
				int node = *it;
				set<int> neighbor_nodes = get_neighbors(this->num_node_samples, node);
				// update phi
				update_phi(node, neighbor_nodes, eps_t);
			}

			// update phi -> pi
			for (int i = 0; i < N; i++){
				double sum = 0;
				for (int k = 0; k < K; k++){
					sum += phi[i][k];
				}
				for (int k = 0; k < K; k++){
					pi[i][k]= phi[i][k]/sum;
					//cout<<pi[i][k];
				}
			}

			update_beta(miniBatch, eps_t);
			if (step_count % 10 == 0){
				this->cal_perplexity(network.held_out_edges);
			}
		}
	}

	set<int> get_neighbors(int sample_size, int nodeId){
		int p = num_node_samples;
		set<int> neighbors_set;

		set<Edge> held_out = network.held_out_edges;
		while (p>0){
			vector<int> nodeList = network.randomSample(N, sample_size*2);
			int size = nodeList.size();
			for (int i = 0; i < size; i++){
				if (p <0){
					break;
				}
				if (nodeList[i] == nodeId){
					continue;
				}
				Edge edge(min(nodeId, nodeList[i]), max(nodeId, nodeList[i]));

				std::set<Edge>::iterator it1;
				it1 = held_out.find(edge);
				if (it1 != held_out.end()){
					continue;
				}
				std::set<int>::iterator it2;
				it2 = neighbors_set.find(nodeList[i]);
				if (it2 != neighbors_set.end()){
					continue;
				}

				neighbors_set.insert(nodeList[i]);
				p -= 1;
			}
		}

		return neighbors_set;
	}

	set<int> nodes_in_batch(set<Edge> &edges){
		set<int> nodes;
		for (std::set<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it){
			 Edge e = *it;
			 nodes.insert(e.u);
			 nodes.insert(e.v);
		}
		return nodes;
	}


	void update_beta(MiniBatch &miniBatch, double eps_t){

		double scale = miniBatch.scale;
		set<Edge> edges = miniBatch.mini_batch_edges;

		// initialize
		double** grads;
		grads = new double*[K];
		for(int k = 0; k <K;k++){
			grads[k] = new double[2]();
			assert(grads[k][0]==0);
			assert(grads[k][1]==0);
		}

		double* sum_theta = new double[K];
		for (int k=0;k<K;k++){
			sum_theta[k]=0;
		}

		for (int k = 0; k < K; k++){
			sum_theta[k] = theta[k][0] + theta[k][1];
		}

		for (std::set<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it){
			Edge e = *it;
			int i = e.u;
			int j = e.v;
			int y = 0;
			std::set<Edge>::iterator it1;
			it1 = network.link_edges.find(e);
			if (it1 != network.link_edges.end()){
				y =1;
			}


			double* probs=new double[K]();
			double sum_pi = 0;
			double sum_prob = 0;
			for (int k = 0; k < K; k++){
				sum_pi += pi[i][k] * pi[j][k];
				probs[k] = pow(beta[k], y)*pow(1-beta[k],1-y)*pi[i][k]*pi[j][k];
				sum_prob += probs[k];
			}
			double prob_0 = pow(epsilon,y)*pow(1-epsilon, 1-y)*(1-sum_pi);
			sum_prob += prob_0;
			for (int k = 0; k < K; k++){
				grads[k][0] += (probs[k]/sum_prob) * (abs(1-y)/theta[k][0]-1/sum_theta[k]);
				grads[k][1] += (probs[k]/sum_prob) * (abs(-y)/theta[k][1]-1/sum_theta[k]);
			}
		}

		double** noise;
		noise = new double*[K];
		for (int k=0;k<K;k++){
			noise[k] = new double[2]();
		}

		gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
		gsl_rng_set(r, time(NULL));
		for (int k = 0; k < K; k++){
			noise[k][0] = gsl_ran_gaussian(r,1);
			noise[k][1] = gsl_ran_gaussian(r,1);

		}
		for (int k = 0; k < K; k++){
			for (int i = 0; i < 2; i++){
				theta[k][i]=abs(theta[k][i] + eps_t/2 * (eta[i] - theta[k][i] +
                        scale* grads[k][i]) + pow(eps_t,0.5)*pow(theta[k][i], 0.5) * noise[k][i]);
			}
		}

		// update beta
		for (int k = 0; k < K; k++){
			beta[k] = theta[k][1]/(theta[k][0]+theta[k][1]);

		}
	}


	void update_phi(int i, set<int> &neighbors, double eps_t){
		int n = neighbors.size();
		double sum_phi = 0;
		for (int k = 0; k < K; k++){
			sum_phi += phi[i][k];
		}

		double* grads=new double[K]();
		double* noise=new double[K]();

		gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
		gsl_rng_set(r, time(NULL));
		for (int k = 0; k < K; k++){
			noise[k] = gsl_ran_gaussian(r,1);
		}
		std::set<Edge>::iterator it1;


		for (std::set<int>::iterator it=neighbors.begin(); it!=neighbors.end(); ++it){
			int j = *it;
			if (i==j){
				continue;
			}

			int y = 0;
			Edge edge(min(i,j),max(i,j));
			it1 = network.link_edges.find(edge);
			if (it1 != network.link_edges.end()){
				y = 1;
			}

			double* probs =new double[K]();
			double sum_prob = 0;
			for (int k = 0; k < K; k++){
				probs[k] = pow(beta[k], y) * pow(1-beta[k], 1-y)*pi[i][k]*pi[j][k];
				probs[k] += pow(epsilon, y)*pow(1-epsilon, 1-y)*pi[i][k]*(1-pi[j][k]);
				sum_prob += probs[k];
			}
			for (int k = 0; k < K; k++){
				grads[k] += (probs[k]/sum_prob)/phi[i][k] - 1.0/sum_phi;
			}
		}

		// update phi
		for (int k = 0; k < K; k++){
			phi[i][k] = abs(phi[i][k] + eps_t/2 * (alpha - phi[i][k] +
                    (N/n)*grads[k]) + pow(eps_t,0.5) * pow(phi[i][k],0.5) * noise[k]);
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
					//cout<<pi[a][k]<<" "<<pi[b][k] <<" "<<beta[k];
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

/*
def run(self):

           # iterate through each node in the mini batch.
           for node in self.__nodes_in_batch(mini_batch):
               noise = np.random.randn(self._K)
               # sample a mini-batch of neighbors
               neighbor_nodes = self.__sample_neighbor_nodes(self.__num_node_sample, node)
               #self._update_phi(node, neighbor_nodes)

               phi_star = update_phi(node, self._step_count, self._epsilon, self._K, self._N,eps_t, self._alpha,
                          self._pi, self._phi, self._beta, noise, len(neighbor_nodes),
                          self._network.get_linked_edges(), neighbor_nodes)
               self._phi[node] = phi_star

           self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]


           # sample (z_ab, z_ba) for each edge in the mini_batch.
           # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
           #z = self.__sample_latent_vars2(mini_batch)
           self.__update_beta(mini_batch, scale)


           if self._step_count % 20 == 0:
               ppx_score = self._cal_perplexity_held_out()
               #print str(ppx_score)
               self._ppxs_held_out.append(ppx_score)

               if self._step_count > 200000:
                   size = len(self._avg_log)
                   ppx_score = (1-1.0/(self._step_count-19000)) * self._avg_log[size-1] + 1.0/(self._step_count-19000) * ppx_score
                   self._avg_log.append(ppx_score)
               else:
                   self._avg_log.append(ppx_score)

               self._timing.append(time.time()-start)

           self._step_count += 1

           if self._step_count % 1000 == 0:
               self._save()

           """
           pr.disable()
           s = StringIO.StringIO()
           sortby = 'cumulative'
           ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
           ps.print_stats()
           print s.getvalue()
           """


    def run1(self):
        while self._step_count < self._max_iteration and not self._is_converged():
            """
            pr = cProfile.Profile()
            pr.enable()
            """
            (mini_batch, scale) = self._network.sample_mini_batch(self._mini_batch_size, "stratified-random-node")

            if self._step_count % 50 == 0:

                ppx_score = self._cal_perplexity_held_out()
                #print "perplexity for hold out set is: "  + str(ppx_score)
                self._ppxs_held_out.append(ppx_score)

            self.__update_pi1(mini_batch, scale)

            # sample (z_ab, z_ba) for each edge in the mini_batch.
            # z is map structure. i.e  z = {(1,10):3, (2,4):-1}
            #z = self.__sample_latent_vars2(mini_batch)
            self.__update_beta(mini_batch, scale)

            """
            pr.disable()
            s = StringIO.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print s.getvalue()
            """
            self._step_count += 1

        print "terminated"




    def __update_pi1(self, mini_batch, scale):

        grads = np.zeros((self._N, self._K))
        counter = np.zeros(self._N)
        phi_star = np.zeros((self._N, self._K))

        for edge in mini_batch:
            a = edge[0]
            b = edge[1]

            y = 0      # observation
            if (min(a, b), max(a, b)) in self._network.get_linked_edges():
                y = 1
            # z_ab
            prob_a = np.zeros(self._K)
            prob_b = np.zeros(self._K)

            for k in range(0, self._K):
                prob_a[k] = self._beta[k]**y*(1-self._beta[k])**(1-y)*self._pi[a][k]*self._pi[b][k]
                prob_a[k] += self._epsilon**y*(1-self._epsilon)**(1-y)*self._pi[a][k]*(1-self._pi[b][k])
                prob_b[k] = self._beta[k]**y*(1-self._beta[k])**(1-y)*self._pi[b][k]*self._pi[a][k]
                prob_b[k] += self._epsilon**y*(1-self._epsilon)**(1-y)*self._pi[b][k]*(1-self._pi[a][k])

            sum_prob_a = np.sum(prob_a)
            sum_prob_b = np.sum(prob_b)

            for k in range(0, self._K):
                grads[a][k] += (prob_a[k]/sum_prob_a)/self._phi[a][k]-1.0/np.sum(self._phi[a])
                grads[b][k] += (prob_b[k]/sum_prob_b)/self._phi[b][k]-1.0/np.sum(self._phi[b])
            #z_ab = self.sample_z_ab_from_edge(y_ab, self._pi[a], self._pi[b], self._beta, self._epsilon, self._K)
            #z_ba = self.sample_z_ab_from_edge(y_ab, self._pi[b], self._pi[a], self._beta, self._epsilon, self._K)
            #print str(grads[a])
            counter[a] += 1
            counter[b] += 1

            #grads[a][z_ab] += 1/self._phi[a][z_ab]
            #grads[b][z_ba] += 1/self._phi[b][z_ba]


        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)

        for i in range(0, self._N):
            noise = np.random.randn(self._K)
            for k in range(0, self._K):
                if counter[i] < 1:
                    phi_star[i][k] = abs((self._phi[i,k]) + eps_t*(self._alpha - self._phi[i,k])+(2*eps_t)**.5*self._phi[i,k]**.5 * noise[k])
                else:
                    phi_star[i][k] = abs(self._phi[i,k] + eps_t * (self._alpha - self._phi[i,k] + \
                               (self._N/counter[i]) * grads[i][k]) \
                                + (2*eps_t)**.5*self._phi[i,k]**.5 * noise[k])

        self._phi = phi_star
        self._pi = self._phi/np.sum(self._phi,1)[:,np.newaxis]


    def _update_phi(self,i, neighbor_nodes):

        n = len(neighbor_nodes)
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)
        """ update phi for node i"""
        sum_phi = np.sum(self._phi[i])
        grads = np.zeros(self._K)
        phi_star = np.zeros(self._K)
        noise = np.random.randn(self._K)

        for j in neighbor_nodes:
            """ for each node j """
            if i == j:
                continue
            y = 0
            if (min(i,j), max(i,j)) in self._network.get_linked_edges():
                y = 1

            probs = np.zeros(self._K)
            for k in range(0, self._K):
                # p(z_ij = k)
                probs[k] = self._beta[k]**y * (1-self._beta[k])**(1-y)*self._pi[i][k]*self._pi[j][k]
                probs[k] += self._epsilon**y *(1-self._epsilon)**(1-y)*self._pi[i][k]*(1-self._pi[j][k])

            sum_prob = np.sum(probs)
            for k in range(0, self._K):
                grads[k] += (probs[k]/sum_prob)/self._phi[i][k] - 1.0/sum_phi

        # update phi
        for k in range(0, self._K):
            phi_star[k] = abs(self._phi[i,k] + eps_t/2 * (self._alpha - self._phi[i,k] + \
                                (self._N/n)*grads[k]) + eps_t**.5*self._phi[i,k]**.5 * noise[k])

        self._phi[i] = phi_star

    def __update_beta(self, mini_batch, scale):
        '''
        update beta for mini_batch.
        '''
        grads = np.zeros((self._K, 2))
        sum_theta = np.sum(self._theta,1)

        # update gamma, only update node in the grad
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)                                  # gradients K*2 dimension                                                         # random noise.


        for edge in mini_batch:
            i = edge[0]
            j = edge[1]
            y = 0
            if edge in self._network.get_linked_edges():
                y = 1

            probs = np.zeros(self._K)
            sum_pi = 0
            for k in range(0, self._K):
                sum_pi += self._pi[i][k]*self._pi[j][k]
                probs[k] = self._beta[k]**y*(1-self._beta[k])**(1-y)*self._pi[i][k]*self._pi[j][k]
            prob_0 = self._epsilon**y*(1-self._epsilon)**(1-y)*(1-sum_pi)
            sum_prob = np.sum(probs) + prob_0
            for k in range(0, self._K):
                grads[k,0] += (probs[k]/sum_prob) * (abs(1-y)/self._theta[k,0]-1/sum_theta[k])
                grads[k,1] += (probs[k]/sum_prob) * (abs(-y)/self._theta[k,1]-1/sum_theta[k])

        noise = np.random.randn(self._K, 2)
        theta_star = copy.copy(self._theta)
        for k in range(0,self._K):
            for i in range(0,2):
                theta_star[k,i] = abs(self._theta[k,i] + eps_t/2 * (self._eta[i] - self._theta[k,i] + \
                                    scale* grads[k,i]) + eps_t**.5*self._theta[k,i] ** .5 * noise[k,i])

        self._theta = theta_star
        #self._theta = theta_star
        # update beta from theta
        temp = self._theta/np.sum(self._theta,1)[:,np.newaxis]
        self._beta = temp[:,1]

    def __update_pi_for_node(self, i, z, n, scale):
        '''
        update pi for current node i.
        '''
        # update gamma, only update node in the grad
        #if self.stepsize_switch == False:
        #    eps_t = (1024+self._step_count)**(-0.5)
        #else:
        eps_t  = self.__a*((1 + self._step_count/self.__b)**-self.__c)

        phi_star = copy.copy(self._phi[i])                              # updated \phi
        phi_i_sum = np.sum(self._phi[i])
        noise = np.random.randn(self._K)                                 # random noise.

        # get the gradients
        grads = [-n * 1/phi_i_sum * j for j in np.ones(self._K)]
        for k in range(0, self._K):
            grads[k] += 1/self._phi[i,k] * z[k]

        # update the phi
        for k in range(0, self._K):
            phi_star[k] = abs(self._phi[i,k] + eps_t/2 * (self._alpha - self._phi[i,k] + \
                                (self._N/n) * grads[k]) + eps_t**.5*self._phi[i,k]**.5 * noise[k])

        self._phi[i] = phi_star

        # update pi
        sum_phi = np.sum(self._phi[i])
        self._pi[i] = [self._phi[i,k]/sum_phi for k in range(0, self._K)]


    def __sample_latent_vars2(self, mini_batch):
        '''
        sample latent variable (z_ab, z_ba) for each pair of nodes. But we only consider 11 different cases,
        since we only need indicator function in the gradient update. More details, please see the comments
        within the sample_z_for_each_edge function.
        '''
        z = {}
        for edge in mini_batch:
            y_ab = 0
            if edge in self._network.get_linked_edges():
                y_ab = 1

            z[edge] = self.__sample_z_for_each_edge(y_ab, self._pi[edge[0]], self._pi[edge[1]], \
                                          self._beta, self._K)

        return z

    def __sample_z_for_each_edge(self, y, pi_a, pi_b, beta, K):
        '''
        sample latent variables z_ab and z_ba
        but we don't need to consider all of the cases. i.e  (z_ab = j, z_ba = q) for all j and p.
        because of the gradient depends on indicator function  I(z_ab=z_ba=k), we only need to consider
        K+1 different cases:  p(z_ab=0, z_ba=0|*), p(z_ab=1,z_ba=1|*),...p(z_ab=K, z_ba=K|*),.. p(z_ab!=z_ba|*)

        Arguments:
            y:        observation [0,1]
            pi_a:     community membership for node a
            pi_b:     community membership for node b
            beta:     community strengh.
            epsilon:  model parameter.
            K:        number of communities.

        Returns the community index. If it falls into the case that z_ab!=z_ba, then return -1
        '''
        p = np.zeros(K+1)
        for k in range(0,K):
            p[k] = beta[k]**y*(1-beta[k])**(1-y)*pi_a[k]*pi_b[k]
        p[K] = 1 - np.sum(p[0:K])

        # sample community based on probability distribution p.
        for k in range(1,K+1):
            p[k] += p[k-1]
        #bounds = np.cumsum(p)
        location = random.random() * p[K]

        # get the index of bounds that containing location.
        for i in range(0, K):
                if location <= p[i]:
                    return i
        return -1


    def __sample_latent_vars(self, node, neighbor_nodes):
        '''
        given a node and its neighbors (either linked or non-linked), return the latent value
        z_ab for each pair (node, neighbor_nodes[i].
        '''
        z = np.zeros(self._K)
        for neighbor in neighbor_nodes:
            y_ab = 0      # observation
            if (min(node, neighbor), max(node, neighbor)) in self._network.get_linked_edges():
                y_ab = 1

            z_ab = self.sample_z_ab_from_edge(y_ab, self._pi[node], self._pi[neighbor], self._beta, self._epsilon, self._K)
            z[z_ab] += 1

        return z
    """
    def __sample_z_ab_from_edge(self, y, pi_a, pi_b, beta, epsilon, K):
        '''
        we need to calculate z_ab. We can use deterministic way to calculate this
        for each k,  p[k] = p(z_ab=k|*) = \sum_{i}^{} p(z_ab=k, z_ba=i|*)
        then we simply sample z_ab based on the distribution p.
        this runs in O(K)
        '''
        p = np.zeros(K)
        for i in range(0, K):
            tmp = beta[i]**y*(1-beta[i])**(1-y)*pi_a[i]*pi_b[i]
            tmp += epsilon**y*(1-epsilon)**(1-y)*pi_a[i]*(1-pi_b[i])
            p[i] = tmp
        # sample community based on probability distribution p.
        bounds = np.cumsum(p)
        location = random.random() * bounds[K-1]

        # get the index of bounds that containing location.
        for i in range(0, K):
                if location <= bounds[i]:
                    return i
        # failed, should not happen!
        return -1
    """    

    def __sample_neighbor_nodes(self, sample_size, nodeId):
        '''
        Sample subset of neighborhood nodes.
        '''
        p = sample_size
        neighbor_nodes = Set()
        held_out_set = self._network.get_held_out_set()
        test_set = self._network.get_test_set()

        while p > 0:
            nodeList = random.sample(list(xrange(self._N)), sample_size * 2)
            for neighborId in nodeList:
                    if p < 0:
                        break
                    if neighborId == nodeId:
                        continue
                    # check condition, and insert into mini_batch_set if it is valid.
                    edge = (min(nodeId, neighborId), max(nodeId, neighborId))
                    if edge in held_out_set or edge in test_set or neighborId in neighbor_nodes:
                        continue
                    else:
                        # add it into mini_batch_set
                        neighbor_nodes.add(neighborId)
                        p -= 1

        return neighbor_nodes

    def __nodes_in_batch(self, mini_batch):
        """
        Get all the unique nodes in the mini_batch.
        """
        node_set = Set()
        for edge in mini_batch:
            node_set.add(edge[0])
            node_set.add(edge[1])
        return node_set

    def _save(self):
        f = open('ppx_mcmc_stochastic.txt', 'wb')
        for i in range(0, len(self._avg_log)):
            f.write(str(math.exp(self._avg_log[i])) + "\t" + str(self._timing[i]) +"\n")
        f.close()



    def sample_z_ab_from_edge(self, y, pi_a, pi_b, beta, epsilon, K):
        p = np.zeros(K)

        tmp = 0.0

        for i in range(0, K):
            tmp = beta[i]**y*(1-beta[i])**(1-y)*pi_a[i]*pi_b[i]
            tmp += epsilon**y*(1-epsilon)**(1-y)*pi_a[i]*(1-pi_b[i])
            p[i] = tmp


        for k in range(1,K):
            p[k] += p[k-1]

        location = random.random() * p[K-1]
        # get the index of bounds that containing location.
        for i in range(0, K):
            if location <= p[i]:
                return i

        # failed, should not happen!
        return -1
        */
