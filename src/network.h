#ifndef NETWORK_H
#define NETWORK_H

#include "mini_batch.h"
#include "edge.h"
#include <fstream>
#include <assert.h>
#include <algorithm>


using namespace std;

class Network
{
public:
	int N;           				// total number of nodes
	int total_num_linked_edges;   	// total number of edges
	double held_out_ratio;  		// percentage of held-out data size
	set<Edge> link_edges;   		// all pair of linked edges
	int num_pieces = 10;         	// it is used for stratified random node sampling. By default 10
	map<int, int> id_to_id; 		// mapping from new ID to new ID.
	map<int, vector<int> > train_link_map;
	set<Edge> held_out_edges;
	int num_held_out_edges;

	Network(){
	}

	Network(double held_out_ratio){
		string line;
		std::string delimiter = "\t";
		ifstream myfile ("/home/liwenzhe/myworkspace/BigLearning/datasets/network.txt");

		if (myfile.is_open())
		{
			int idx = 0;
			while ( getline (myfile,line) )
			{
				size_t pos = 0;
			    std::string token;
			    vector<int> nodes;
			    while ((pos = line.find(delimiter)) != std::string::npos) {
			      	token = line.substr(0, pos);
			      	nodes.insert(nodes.end(), 1, atoi(token.c_str()));
			      	line.erase(0, pos + delimiter.length());
			    }
			    nodes.insert(nodes.end(),1, atoi(line.c_str()));
			    assert(nodes.size() == 2);
			    cout<<nodes[0]<<" "<<nodes[1];

			    for (unsigned int i = 0; i < nodes.size(); i++){
			    	int nodeId = nodes[i];
			    	if (id_to_id.find(nodeId) == id_to_id.end()){
			    		// not found, to do something.
			    		id_to_id[nodeId] = idx;
			    		++idx;
			    	}
			    }
			}
			myfile.close();
		}

		ifstream myfile2 ("/home/liwenzhe/myworkspace/BigLearning/datasets/network.txt");
		if (myfile2.is_open())
		{
			while ( getline (myfile2,line) )
			{
				size_t pos = 0;
				std::string token;
				vector<int> nodes;
				while ((pos = line.find(delimiter)) != std::string::npos) {
					token = line.substr(0, pos);
					nodes.insert(nodes.end(), 1, atoi(token.c_str()));
					line.erase(0, pos + delimiter.length());
				}
				nodes.insert(nodes.end(),1, atoi(line.c_str()));
				assert(nodes.size() == 2);
				int first = id_to_id[nodes[0]];
				int second = id_to_id[nodes[1]];
				link_edges.insert(Edge(first, second));
			}
		}
		myfile2.close();

		// initialize variables.
		N = id_to_id.size();
		total_num_linked_edges = link_edges.size();
		num_held_out_edges = total_num_linked_edges * held_out_ratio;

		init_train_link_map();
		init_held_out_set();

	}

	void init_train_link_map(){
		// group neighbor nodes for given node.
		for (int i = 0; i < N; ++i){
			vector<int> neighbors;
			train_link_map[i] = neighbors;
		}

		for (std::set<Edge>::iterator it=link_edges.begin(); it!=link_edges.end(); ++it){
	  		Edge e = *it;
	  		vector<int> neighbors1 = train_link_map[e.u];
	  		neighbors1.insert(neighbors1.end(), 1, e.v);
	  		train_link_map[e.u] = neighbors1;

	  		vector<int> neighbors2 = train_link_map[e.v];
	  		neighbors2.insert(neighbors2.end(),1, e.u);
	  		train_link_map[e.v] = neighbors2;
	  	}
	}

	void init_held_out_set(){
		int p = num_held_out_edges/2;
		if (total_num_linked_edges < p){
			cout << "There are not enough linked edges that can sample from. \
                    please use smaller held out ratio"<<endl;
		}

		// sample link edges.
		vector<int> sampleIndexs = randomSample(total_num_linked_edges, p);

		int count = sampleIndexs.size();

		int idx = 0;
		int counter = 0;  // track the sampleIndexs.
		for (std::set<Edge>::iterator it=this->link_edges.begin(); it!=this->link_edges.end(); ++it){
			Edge e = *it;
			if (counter == count){
				// reach the end
				break;
			}

			if (sampleIndexs[counter] == idx){
				// remove e from the train_link_map
				std::vector<int> vec1 = train_link_map[e.u];
				vec1.erase(std::remove(vec1.begin(), vec1.end(), e.v), vec1.end());
				std::vector<int> vec2 = train_link_map[e.v];
				vec2.erase(std::remove(vec2.begin(), vec2.end(), e.u), vec2.end());
				// put into the held out
				held_out_edges.insert(e);
				counter += 1;
			}
			idx += 1;
		}

		p = num_held_out_edges/2;
		// sample non-link edges.
		while(p>0){
			Edge edge = sample_non_link_edge();
			held_out_edges.insert(edge);
			p -= 1;
		}
	}

	Edge sample_non_link_edge(){
		while (1){
			int firstIdx = rand()% N;
			int secondIdx = rand()%N;

			if (firstIdx == secondIdx){
				continue;
			}
			Edge edge(min(firstIdx, secondIdx),max(firstIdx, secondIdx));
			std::set<Edge>::iterator it1;
			it1 = this->link_edges.find(edge);
			if (it1 != link_edges.end()){
				continue;
			}
			it1 = this->held_out_edges.find(edge);
			if (it1 != this->held_out_edges.end()){
				continue;
			}

			return edge;
		}
	}

	MiniBatch sample_mini_batch(){


		int nodeId = rand()%N;
//out<<"id:  "<<nodeId;
		int flag = rand()%2;
	//	cout<<"flag: "<<flag<<endl;
		set<Edge> mini_batch_set;
		if (flag == 0){   // sample non-link edges
			int mini_batch_size = N/this->num_pieces;
			int p = mini_batch_size;
			while (p >0){
				vector<int> samples = this->randomSample(N,2*mini_batch_size);
				int size = samples.size();
				for (int i = 0; i < size; i++){
					if(p<0){
						break;
					}
					if (samples[i] == nodeId){
						continue;
					}

					Edge edge(min(nodeId, samples[i]),max(nodeId, samples[i]));
					std::set<Edge>::iterator it1;
					it1 = this->link_edges.find(edge);
					if (it1 != link_edges.end()){
						continue;
					}
					it1 = this->held_out_edges.find(edge);
					if (it1 != held_out_edges.end()){
						continue;
					}
					it1 = mini_batch_set.find(edge);
					if (it1 != mini_batch_set.end()){
						continue;
					}

					mini_batch_set.insert(edge);
					p -= 1;
				}
			}

			//cout<<mini_batch_set.size();
			return MiniBatch(mini_batch_set, N * this->num_pieces);
		}else{
			// return all the linked edges.
			vector<int> neighbors = this->train_link_map[nodeId];
			int n = neighbors.size();
			for (int i = 0; i < n; i++){
				Edge edge(min(nodeId, neighbors[i]),max(nodeId, neighbors[i]));
				mini_batch_set.insert(edge);
			}

			return MiniBatch(mini_batch_set, N);
		}
	}

	vector<int> randomSample(int n, int m){
		// select m sample

		vector<int> indexs(n,0);
		for (int i = 0; i < n; i++){
			indexs[i] = i;
		}
		for (int i = 0; i < m; i++){
			int replaced = rand()%(n-i)+i;
			//cout<<replaced<<endl;
			int tmp = indexs[i];
			indexs[i] = indexs[replaced];
			indexs[replaced] = tmp;
		}

		vector<int> result(m,0);
		for (int i = 0; i < m; i++)
			result[i] = indexs[i];
		std::sort(result.begin(), result.end());

		return result;
	}
};

#endif
