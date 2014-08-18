#ifndef MINI_BATCH
#define MINI_BATCH

#include <set>

using namespace std;

class MiniBatch{
public:
	set<Edge> mini_batch_edges;
	double scale;

	MiniBatch(set<Edge> mini_batch_edges, double scale){
		this->mini_batch_edges = mini_batch_edges;
		this->scale = scale;
	}
};

#endif
