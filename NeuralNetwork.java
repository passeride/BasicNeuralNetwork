

public class NeuralNetwork {

	public double [][] x = new double[][]{
		{0,0},
		{0,1}, 
		{1,0},
		{1,1}, 
	};
	
	public double [][] y = new double[][]{
		{0},
		{1}, 
		{1},
		{0}, 
	};
	
	public double [][] syn0;
	public double [][] syn1;
	
	public int hiddenLayers = 1;
	public int nodesHLay = 4;
	public double learningRate = 0.0005;
	
	public NeuralNetwork(){
		populateSynapeses();
		
		for(int i = 0; i < 10000000; i++){
			double[][] l0 = x;
			double[][] l1 = sigmoid(matrixDotProduct(l0, syn0));
			double[][] l2 = sigmoid(matrixDotProduct(l1, syn1));;
			double[][] l2_error = matrixSubtraction(y, l2);
			double[][] l2_delta = matrixMultiplication(l2_error, sigmoidPrime(l2));
			double[][] l1_error = matrixDotProduct(l2_delta, transposeMatrix(syn1));
			double[][] l1_delta = matrixMultiplication(l1_error, sigmoidPrime(l1));
			
			// update weights
			l2_delta = matrixMultiplicationSingel(l2_delta, learningRate);
			double[][] l2_deltaDot = matrixDotProduct(transpose(l1), l2_delta);
			
			syn1 = matrixAddition(l2_deltaDot, syn1);
			
			l1_delta = matrixMultiplicationSingel(l1_delta, learningRate);
			double[][] l1_deltaDot = matrixDotProduct(transpose(l0), l1_delta);
			syn0 = matrixAddition(l1_deltaDot, syn0);
			
			if(i % 100000 == 0){
				System.out.printf("\r\n Error is : %.4f ", Math.abs(matrixMean(l2_error)));
				printSideBySide(l2, y);
			}
		}
	}
	

	void printSideBySide(double[][] l2, double[][] out){
		for(int i = 0; i < l2.length; i++){
			System.out.println("");
			for(int j = 0; j < l2[i].length; j++){
				System.out.printf("%.2f : %.2f", l2[i][j], out[i][j]);
			}
		}
		System.out.println("");
	}
	
	public double matrixMean(double[][] m){
		double ret = 0.0;
		int iterations = 0; 
		
		for(int i = 0; i < m.length; i++){
			for(int j = 0; j < m[i].length; j++){
				ret += m[i][j];
				iterations ++;
			}
		}
		
		return ret / iterations;
	}
	
    public double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[j][i] = a[i][j];
        return b;
    }
	
	public double[][] matrixMultiplicationSingel(double[][] x, double multiplicator){
		double[][] ret = new double[x.length][x[0].length];
		for(int i = 0; i < ret.length; i++){
			for(int j = 0; j < ret[i].length; j++){
				ret[i][j] = x[i][j] * multiplicator;
			}
		}
		return ret;
	}
	
    public static double[][] transposeMatrix(double [][] m){
        double[][] temp = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }
	
	public double[][] matrixSubtraction(double[][] x, double[][] y){
		int xRows = x.length;
		int xColumns = x[0].length;
		int yRows = y.length;
		int yColumns = y[0].length;
		
		if(xColumns != yRows){
//			throw new IllegalArgumentException("X:Rows: " + xRows + " did not match yColumns " + yColumns);
		}
		
		double[][] ret = new double[xRows][yColumns];
		
		for(int i = 0; i < ret.length; i++){
			for(int j = 0; j < ret[i].length; j++){
				ret[i][j] = x[i][j] - y[i][j];
			}
		}	
		
		return ret;
	}
	
	public double[][] matrixMultiplication(double[][] x, double[][] y){
		int xRows = x.length;
		int xColumns = x[0].length;
		int yRows = y.length;
		int yColumns = y[0].length;
		
		if(xColumns != yRows){
//			throw new IllegalArgumentException("X:Rows: " + xRows + " did not match yColumns " + yColumns);
		}
		
		double[][] ret = new double[xRows][yColumns];
		
		for(int i = 0; i < ret.length; i++){
			for(int j = 0; j < ret[i].length; j++){
				ret[i][j] = x[i][j] * y[i][j];
			}
		}	
		
		return ret;
	}
	
	public double[][] matrixAddition(double[][] x, double[][] y){
		int xRows = x.length;
		int xColumns = x[0].length;
		int yRows = y.length;
		int yColumns = y[0].length;
		
		if(xColumns != yRows){
//			throw new IllegalArgumentException("X:Rows: " + xRows + " did not match yColumns " + yColumns);
		}
		
		double[][] ret = new double[xRows][yColumns];
		
		for(int i = 0; i < ret.length; i++){
			for(int j = 0; j < ret[i].length; j++){
				ret[i][j] = x[i][j] + y[i][j];
			}
		}	
		
		return ret;
	}
	
	void printMatrix(double[][] x){
		for(int i = 0; i < x.length; i++){
			System.out.println("");
			for(int j = 0; j < x[i].length; j++){
				System.out.print(" " + x[i][j]);
			}
		}
		System.out.println("");
	}
	
	public double[][] matrixDotProduct(double[][] x, double[][] y){
		int xRows = x.length;
		int xColumns = x[0].length;
		int yRows = y.length;
		int yColumns = y[0].length;
		
		if(xColumns != yRows){
			throw new IllegalArgumentException("X:Rows: " + xRows + " did not match yColumns " + yColumns);
		}
		
		double[][] ret = new double[xRows][yColumns];
		
		for(int i = 0; i < ret.length; i++){
			for(int j = 0; j < ret[i].length; j++){
				ret[i][j] = 0.000000;
			}
		}
		
		for(int i = 0; i < ret.length; i++){
			for(int j = 0; j < ret[i].length; j++){
				for(int k = 0; k < xColumns; k++){
					ret[i][j] += x[i][k] * y[k][j];
				}
			}
		}
		
		
		return ret;
	}
	
    public double sigmoid(double x) {
    	return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }
    
    public double[][] sigmoid(double[][] x) {
    	double[][] ret = new double[x.length][x[0].length];
    	for(int i = 0; i < ret.length; i++){
    		for(int j = 0; j < ret[i].length; j++){
    			ret[i][j] = sigmoid(x[i][j]);
    		}
    	}
    	return ret;
    }
    
    public double sigmoidPrime(double x){
    	return sigmoid(1 - sigmoid(x));
    }
    
    public double[][] sigmoidPrime(double[][] x){    	
    	double[][] ret = new double[x.length][x[0].length];
    	for(int i = 0; i < ret.length; i++){
    		for(int j = 0; j < ret[i].length; j++){
    			ret[i][j] = sigmoidPrime(x[i][j]);
    		}
    	}
    	return ret;
	}
    
    public double getRandom(){
    	return Math.random() * 40 - 20;
    }
	
    public void populateSynapeses(){
    	
    	syn0 = new double[x[0].length][];
		
    	
    	for(int i = 0; i < x[0].length; i++){
    		syn0[i] = new double[nodesHLay];
    		for(int j = 0; j < nodesHLay; j++){
    			syn0[i][j] = getRandom();
    			
    		}
    	}
    	
    	syn1 = new double[nodesHLay][];
    	for(int i = 0; i < nodesHLay; i++){
    		syn1[i] = new double[y[0].length];
    		for(int j = 0; j < y[0].length; j++){
    			syn1[i][j] = getRandom();
    		}
    	}
    	
    }
    
}
