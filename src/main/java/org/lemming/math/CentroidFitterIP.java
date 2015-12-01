package org.lemming.math;

import ij.gui.Roi;
import ij.process.ImageProcessor;

/**
 * Calculating centroids on a {@link #ImageProcessor}
 * 
 * @author Ronny Sczech
 *
 */
public class CentroidFitterIP {
	
	private static double defaultSigma = 1.5;
	
	public static double[] fitThreshold(ImageProcessor ip_, Roi roi){
		double[] centroid = new double[2];
		int rwidth = (int) roi.getFloatWidth();
		int rheight = (int) roi.getFloatHeight();
		int xstart = (int) roi.getXBase();
		int ystart = (int) roi.getYBase();
		
		// Copy the ImageProcessor and carry on threshold
		ImageProcessor ip = ip_.duplicate();
		ip.setRoi(roi);
		ip.autoThreshold();
		
		// Find centroid
		double sum = 0;
		for(int i=ystart;i<rheight+ystart;i++){
			for(int j=xstart;j<rwidth+xstart;j++){
				if(ip.get(j, i)>0){
					centroid[0] += j;
					centroid[1] += i;
					sum ++;
				}
			}
		}
		centroid[0] = centroid[0]/sum;
		centroid[1] = centroid[1]/sum; 
		
		return centroid;
	}
	
	public static double[] fitCentroid(ImageProcessor ip_, Roi roi){
		double[] centroid = new double[2];
		int rwidth = (int) roi.getFloatWidth();
		int rheight = (int) roi.getFloatHeight();
		int xstart = (int) roi.getXBase();
		int ystart = (int) roi.getYBase();
		
		// Copy the ImageProcessor and carry on threshold
		ImageProcessor ip = ip_.duplicate();
		
		// Find centroid
		double sum = 0;
		for(int i=ystart;i<rheight+ystart;i++){
			for(int j=xstart;j<rwidth+xstart;j++){
				if(ip.get(j, i)>0){
					centroid[0] += j*ip.get(j, i);
					centroid[1] += i*ip.get(j, i);
					sum += ip.get(j, i);
				}
			}
		}
		centroid[0] = centroid[0]/sum;
		centroid[1] = centroid[1]/sum; 
		
		return centroid;
	}
	
	public static double[] fitCentroidandWidth(ImageProcessor ip, Roi roi, int threshold){
		double[] centroid = new double[4];
		int rwidth = (int) roi.getFloatWidth();
		int rheight = (int) roi.getFloatHeight();
		int xstart = (int) roi.getXBase();
		int ystart = (int) roi.getYBase();
		
		int thrsh = threshold;
		
		// Find centroid and widths
		int s = 0;
		double sum = 0;
		for(int i=xstart;i<rwidth+xstart;i++){
			for(int j=ystart;j<rheight+ystart;j++){
				s = ip.get(i, j);
				if(s>thrsh){
					centroid[0] += i*s;
					centroid[1] += j*s;
					sum += s;
				}
			}
		}
		centroid[0] = centroid[0]/sum;
		centroid[1] = centroid[1]/sum; 
		
		if(Double.isNaN(centroid[0]))
			centroid[0] = xstart+rwidth/2;
		
		if(Double.isNaN(centroid[1]))
			centroid[1] = ystart+rheight/2;
		
		double sumstd=0, stdx=0, stdy=0;
		for(int i=0;i<rheight;i++){
			for(int j=0;j<rwidth;j++){
				if(ip.get(j, i)>thrsh){
					sumstd += ip.get(j, i);
					stdx += ip.get(j, i)*(xstart+j-centroid[0])*(xstart+j-centroid[0]);
					stdy += ip.get(j, i)*(ystart+i-centroid[1])*(ystart+i-centroid[1]);
				}
			}
		}

		stdx /= sumstd;
		stdy /= sumstd;
		stdx = Math.sqrt(stdx);
		stdy = Math.sqrt(stdy);
				
		if(Double.isNaN(stdx)){
			centroid[2] = defaultSigma ;
		} else {
			centroid[2] = stdx;
		}

		if(Double.isNaN(stdy)){
			centroid[3] = defaultSigma;
		} else {
			centroid[3] = stdy;
		}
		
		return centroid;
	}

}
