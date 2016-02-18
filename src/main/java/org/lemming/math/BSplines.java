package org.lemming.math;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;

public class BSplines {

	private PolynomialSplineFunction fwx;
	private PolynomialSplineFunction fwy;

	public BSplines(double[] z, double[] wx, double[] wy) {
		SplineInterpolator interpolator = new SplineInterpolator();
		fwx = interpolator.interpolate(z, wx);
		fwy = interpolator.interpolate(z, wx);
	}
	
	private String doubleArrayToString(double[] array){
		String result ="";
		for (int num=0; num<array.length;num++)
			result += array[num] + ",";
		result = result.substring(0, result.length()-1);
		return result;
	}
	
	private double[] stringToDoubleArray(String line){
		String[] s = line.split(",");
		double[] result = new double[s.length];
		for (int n=0;n<s.length;n++)
			result[n]=Double.parseDouble(s[n].trim());
		return result;
	}
	
	public void saveAsCSV(String path){
		final PolynomialFunction[] polynomsX = fwx.getPolynomials();
		final double[] knotsX = fwx.getKnots();
		final PolynomialFunction[] polynomsY = fwy.getPolynomials();
		final double[] knotsY = fwy.getKnots();
		
		try {
			FileWriter w = new FileWriter(new File(path));
			w.write(doubleArrayToString(knotsX)+"\n");
			for (int i=0; i<polynomsX.length;i++)
				w.write(doubleArrayToString(polynomsX[i].getCoefficients())+"\n");
			w.write("--\n");
			w.write(doubleArrayToString(knotsY)+"\n");
			for (int i=0; i<polynomsY.length;i++)
				w.write(doubleArrayToString(polynomsY[i].getCoefficients())+"\n");
			
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public PolynomialSplineFunction[] readCSV(String path){
		PolynomialSplineFunction[] functions = new PolynomialSplineFunction[2];
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));
			String line=br.readLine();
			final double[] knotsX = stringToDoubleArray(line);
			PolynomialFunction[] polynomsX = new PolynomialFunction[knotsX.length-1];
			for (int n=0;n<polynomsX.length;n++){
				line=br.readLine();
				polynomsX[n]=new PolynomialFunction(stringToDoubleArray(line));
			}
			
			if (br.readLine()!="--") System.err.println("Corrupt File!");
			line=br.readLine();
			final double[] knotsY = stringToDoubleArray(line);
			PolynomialFunction[] polynomsY = new PolynomialFunction[knotsY.length-1];
			for (int n=0;n<polynomsY.length;n++){
				line=br.readLine();
				polynomsY[n]=new PolynomialFunction(stringToDoubleArray(line));
			}

			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return functions;
	}

}
