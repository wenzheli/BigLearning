//============================================================================
// Name        : BigLearning.cpp
// Author      : wenzhe
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "test.h"
#include "gsl/gsl_sf_psi.h"
#include "gsl/gsl_sort_vector.h"
#include <set>
#include <vector>
#include <sstream>
#include "edge.h"
#include <map>
#include <assert.h>
#include <time.h>
#include "network.h"
#include "gsl/gsl_randist.h"
#include "MCMCSampler.h"
#include "args.h"
using namespace std;

int main() {

	srand(time(NULL));
	clock_t t1,t2;

	t1=clock();
	Args arg;
	arg.alpha = 0.01;
	arg.eta0 = 1;
	arg.eta1 =1;
	arg.K = 10;
	arg.epsilon = 0.0000001;
	arg.max_iteration = 1000;
	arg.a = 0.01;
	arg.b =1024;
	arg.c = 0.55;

	Network net(0.01);
	set<Edge> edges = net.link_edges;
	for (std::set<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it){
		Edge e = *it;
		cout<<e.u<<" "<<e.v<<endl;
	}

	cout<<"**************************";
	map<int, vector<int> > train_map = net.train_link_map;
	vector<int> neighbors1 = train_map[0];
	for(int i=0;i<neighbors1.size();i++){
		cout<<neighbors1[i]<<endl;
	}
	cout<<"**************************";
	neighbors1 = train_map[1];
	for(int i=0;i<neighbors1.size();i++){
		cout<<neighbors1[i]<<endl;
	}

	MCMCSampler sampler(arg, net);
	sampler.run();

	cout<<"time elapsed:";
	t2=clock();
	float diff= ((float)t2-(float)t1);
	float seconds = diff/CLOCKS_PER_SEC;
	cout<<seconds<<endl;

	/*

	gsl_rng * r = gsl_rng_alloc (gsl_rng_taus);
	gsl_rng_set(r, time(NULL));

	double p1 = gsl_ran_gamma(r,100,0.01);
	int* p;
	p = new int[10];
	p[0] = 1;
	p[1] = 2;
	cout<<p[3];
	clock_t t1,t2;
	 t1=clock();
	    //code goes here

	vector<int> samples;
	Network n1(0.1);

	set<Edge> edges = n1.held_out_edges;
	for (std::set<Edge>::iterator it=edges.begin(); it!=edges.end(); ++it){
		  		Edge e = *it;
		  		cout<<e.to_string()<<endl;
		  	}

	t2=clock();
	float diff= ((float)t2-(float)t1);
	float seconds = diff/CLOCKS_PER_SEC;
	cout<<seconds<<endl;

	cout<<samples[0]<<endl;
	cout<<samples[1]<<endl;
	cout<<samples[2]<<endl;
	cout<<samples[3]<<endl;
	cout<<samples[4]<<endl;



	vector<int> num;
	cout<<num.size();

	map<int, vector<int> > myMap;

	vector<int> aaaaa(10,1);

	aaaaa.insert(aaaaa.end(),4,3);
    cout<<aaaaa.size();



    myMap[1] = aaaaa;

    vector<int> bb(10,1);
    myMap[2] = bb;

	std::set<Edge> edge;

	edge.insert(Edge(1,2)); // <-- (1,2) can be contained.
	edge.insert(Edge(1,2)); // <-- (1,2) doesn't have to be contained.
	edge.insert(Edge(2,1)); // <-- (1,2) doesn't have to be contained.
	edge.insert(Edge(1,3)); // <-- (1,2) doesn't have to be contained.
	edge.insert(Edge(1,4)); // <-- (1,2) doesn't have to be contained.
	edge.insert(Edge(1,5)); // <-- (1,2) doesn't have to be contained.
	edge.insert(Edge(4,1)); // <-- (1,2) doesn't have to be contained.
	edge.insert(Edge(5,1)); // <-- (1,2) doesn't have to be contained.
	//myEdgeSet.insert(edge2);
	//	myEdgeSet.insert(edge3);
	  cout<< edge.size();

	  	for (std::set<Edge>::iterator it=edge.begin(); it!=edge.end(); ++it){
	  		Edge e = *it;
	  		cout<<e.to_string()<<endl;
	  	}
	   if (Edge(1,2) == Edge(1,2)){
		   cout<<"same";
	   }
		std::set<Edge>::iterator it1;
		it1 = edge.find(Edge(1,6));
		if (it1 != edge.end()){
			cout<<"found";
		}





	 set<int> aa;
	 aa.insert(1);
	 aa.insert(2);



	  string aaa = "adfadfadsf";
	 using namespace std;
	   //cout << "The sum of 3 and 4 is " << add(3, 4) << endl;

	    int a[] = { 1, 2, 3, 4 };
	      int b[4];
	      double y = gsl_sf_psi(0.1);
	      cout<<y;

	      // This is the preferred method to copy raw arrays in C++ and works with all types that can be copied:
	      std::copy(a, a+4, b);

	      cout<<b[0];

	      cout<<"******************"<<endl;
	      string line;
	      std::string delimiter = "\t";
	      ifstream myfile ("/home/liwenzhe/myworkspace/BigLearning/datasets/network.txt");
	      if (myfile.is_open())
	      		{
	      		   while ( getline (myfile,line) )
	      		   {

	      			 size_t pos = 0;
	      			 	      std::string token;
	      			 	      vector<int> nodes;
	      			 	      while ((pos = line.find(delimiter)) != std::string::npos) {
	      			 	          token = line.substr(0, pos);
	      			 	          //std::cout << token << std::endl;
	      			 	          nodes.insert(nodes.end(), 1, atoi(token.c_str()));
	      			 	       line.erase(0, pos + delimiter.length());
	      			 	      }
	      			 	      //std::cout << atoi(line.c_str()) << std::endl;
	      			 	      nodes.insert(nodes.end(),1, atoi(line.c_str()));
	      			 	      assert(2 == nodes.size());
	      		   }
	      		   myfile.close();
	      		}

*/

}
