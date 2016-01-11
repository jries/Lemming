package org.lemming.math;

import java.awt.Rectangle;
import java.util.ArrayList;
import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.modules.Fitter;
import org.lemming.pipeline.Localization;
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
	
	public static float[] fitThreshold(ImageProcessor ip, Roi roi, int threshold){
		float[] centroid = new float[3];
		int rwidth = roi.getBounds().width;
		int rheight = roi.getBounds().height;
		int xstart = roi.getBounds().x;
		int ystart = roi.getBounds().y;
		
		ip.setRoi(roi);
		
		// Find centroid
		float sum = 0;
		for(int i=ystart;i<rheight+ystart;i++){
			for(int j=xstart;j<rwidth+xstart;j++){
				if(ip.get(j, i)>threshold){
					centroid[0] += j;
					centroid[1] += i;
					sum ++;
				}
			}
		}
		centroid[0] = centroid[0]/sum;
		centroid[1] = centroid[1]/sum; 
		
		if(Double.isNaN(centroid[0]))
			centroid[0] = xstart+rwidth/2;
		
		if(Double.isNaN(centroid[1]))
			centroid[1] = ystart+rheight/2;
		
   		centroid[2] = ip.get(Math.round(centroid[0]),Math.round(centroid[1]));
		return centroid;
	}
	
	public static List<Element> fit(List<Element> sliceLocs, ImageProcessor ip, long size, float pixelDepth) {
		final List<Element> found = new ArrayList<>();
		final Rectangle imageRoi = ip.getRoi();
		
		for (Element el : sliceLocs) {
			final Localization loc = (Localization) el;
			
			final double x = loc.getX().doubleValue()/pixelDepth;
			final double y = loc.getY().doubleValue()/pixelDepth;
			final Roi origroi = new Roi(x - size, y - size, 2 * size + 1, 2 * size + 1);
			final Roi roi = Fitter.cropRoi(imageRoi, origroi.getBounds());
			final float[] res = fitCentroid(ip, roi);
		
			found.add(new Localization(res[0]*pixelDepth, res[1]*pixelDepth, res[2], 1L));
		}

		return found;
	}
	
	public static float[] fitCentroid(ImageProcessor ip, Roi roi){
		float[] centroid = new float[3];
		int rwidth = roi.getBounds().width;
		int rheight = roi.getBounds().height;
		int xstart = roi.getBounds().x;
		int ystart = roi.getBounds().y;
		
		// Find centroid
		float sum = 0;
		int s = 0;
		for(int i=ystart;i<rheight+ystart;i++){
			for(int j=xstart;j<rwidth+xstart;j++){
				s = ip.get(i, j);
				if(ip.get(j, i)>0){
					centroid[0] += j*s;
					centroid[1] += i*s;
					sum += s;
				}
			}
		}
		centroid[0] = centroid[0]/sum;
		centroid[1] = centroid[1]/sum; 
		
		if(Float.isNaN(centroid[0]))
			centroid[0] = xstart+rwidth/2;
		
		if(Float.isNaN(centroid[1]))
			centroid[1] = ystart+rheight/2;
		
		centroid[2] = ip.get(Math.round(centroid[0]),Math.round(centroid[1]));
		return centroid;
	}
	
	public static double[] fitCentroidandWidth(ImageProcessor ip, Roi roi, int threshold){
		double[] centroid = new double[4];
		int rwidth = roi.getBounds().width;
		int rheight = roi.getBounds().height;
		int xstart = roi.getBounds().x;
		int ystart = roi.getBounds().y;
				
		// Find centroid and widths
		int s = 0;
		double sum = 0;
		for(int i=xstart;i<rwidth+xstart;i++){
			for(int j=ystart;j<rheight+ystart;j++){
				s = ip.get(i, j);
				if(s>threshold){
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
				if(ip.get(j, i)>threshold){
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
