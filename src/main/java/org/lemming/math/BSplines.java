package org.lemming.math;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import javax.swing.JFrame;
import javax.swing.JPanel;

import org.apache.commons.math3.analysis.interpolation.SplineInterpolator;
import org.apache.commons.math3.analysis.polynomials.PolynomialFunction;
import org.apache.commons.math3.analysis.polynomials.PolynomialSplineFunction;
import org.apache.commons.math3.fitting.PolynomialCurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoints;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.lemming.tools.LemmingUtils;


class BSplines {

	private PolynomialSplineFunction fwx;
	private PolynomialSplineFunction fwy;
	private double[] zgrid;
	private final JFrame plotWindow;
	private double[] bestE;
	private double[] Wx;
	private double[] Wy;
	private static final int numKnots = 21;
	
	BSplines(){
		plotWindow = new JFrame();
		plotWindow.setPreferredSize(new Dimension(800,600));
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		plotWindow.setLocation(dim.width/2-400, dim.height/2-300);
		plotWindow.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	}

	void init(double[] z, double[] Wx, double[] Wy, double e[]) {
		zgrid=z;
		this.Wx = Wx;							// width in x of the PSF
		this.Wy = Wy;							// width in y of the PSF
		final SplineInterpolator interpolator = new SplineInterpolator();
		double[] kz = new double[numKnots];
		double[] kwx = new double[numKnots];
		double[] kwy = new double[numKnots];
		calculateKnots(z, Wx, kz, kwx);
		calculateKnots(z, Wy, kz, kwy);
		fwx = interpolator.interpolate(kz, kwx);
		fwy = interpolator.interpolate(kz, kwy);
		final PolynomialCurveFitter fitter = PolynomialCurveFitter.create(0).withStartPoint(new double[] {-1,1,-1,1,-1,1,-1,1,-1,1,-1,1});
		final WeightedObservedPoints obs = new WeightedObservedPoints();
		for (int i=0;i<z.length;i++)
			obs.add(z[i],e[i]);
		bestE = fitter.fit(obs.toList());
	}
	
	private void calculateKnots(double[] z, double[] w, double[] kz, double[] kw){
		float jump = (float)z.length / (numKnots-1);
		for (int i=0;i<numKnots-1;i++){
			int index = Math.round(i*jump);
			kz[i] = z[index];
			kw[i] = w[index];
		}
		kz[numKnots-1]=z[z.length-1];
		kw[numKnots-1]=w[w.length-1];
	}

	private double findIntersection(){
		int index = 0;
		double diff;
		double max = Double.MAX_VALUE;
		for (int i=0; i<zgrid.length;i++){
			diff = Math.abs(fwx.value(zgrid[i]) - fwy.value(zgrid[i]));
			if (diff<max){ 
				index = i;
				max=diff;
			}
		}
		return zgrid[index];
	}
	
	void saveAsCSV(String path){
		final PolynomialFunction[] polynomsX = fwx.getPolynomials();
		final double[] knotsX = fwx.getKnots();
		final PolynomialFunction[] polynomsY = fwy.getPolynomials();
		final double[] knotsY = fwy.getKnots();
		final double zStep=Math.abs(zgrid[zgrid.length-1]-zgrid[0])/zgrid.length;
		
		try {
			FileWriter w = new FileWriter(new File(path));
			w.write(LemmingUtils.doubleArrayToString(knotsX)+"\n");
			for (PolynomialFunction aPolynomsX : polynomsX)
				w.write(LemmingUtils.doubleArrayToString(aPolynomsX.getCoefficients()) + "\n");
			w.write("--\n");
			w.write(LemmingUtils.doubleArrayToString(knotsY)+"\n");
			for (PolynomialFunction aPolynomsY : polynomsY)
				w.write(LemmingUtils.doubleArrayToString(aPolynomsY.getCoefficients()) + "\n");
			w.write("--\n");
			w.write(Double.toString(findIntersection())+"\n");
			w.write(Double.toString(zStep)+"\n");
			w.write(LemmingUtils.doubleArrayToString(zgrid)+"\n");
			w.write(LemmingUtils.doubleArrayToString(bestE)+"\n");
			w.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	private static double[] valuesWith(double z[], PolynomialSplineFunction function) {
		double[] values = new double[z.length];
		for (int i = 0; i < z.length; ++i) 
			values[i] = function.value(z[i]);
		return values;
	}
	
	///////////////////////////////////////// Plot
	public void plotPoints(double[] z, double[] Wx, double[] Wy, String title){
		if(Wx.length > 0 && Wy.length>0){
			createXYDots(createDataSet(z, Wx, "Wx"), createDataSet(z, Wy, "Wy"),"Z (nm)", "Width", title);
		}
	}	
	
	public void plot(double[] W, String title){
		if(W.length > 0){
			PolynomialFunction function = new PolynomialFunction(bestE);
			double[] curveE = new double[zgrid.length];
			for (int i = 0; i < W.length; ++i) {
				curveE[i] = function.value(zgrid[i]);
			}
			createXYDotsAndLines(createDataSet(zgrid, W, "calculated"), createDataSet(zgrid, curveE, "fitted"),"Z (nm)", "e", title);
		}
	}	
	
	public void plotWxWyFitCurves(){
		double[] curveWx = valuesWith(zgrid,fwx); 
		double[] curveWy = valuesWith(zgrid,fwy);
		if(Wx.length > 0 && Wy.length>0){
			createXYDotsAndLines(createDataSet(zgrid, Wx, "Width in X", Wy, "Width in Y"),
				createDataSet(zgrid, curveWx, "Fitted width in X", curveWy, "Fitted width in Y"), "Z (nm)", "Width", "Width of Elliptical Gaussian");
		}
	}

	private static XYDataset createDataSet(double[] X, double[] Y1, String nameY1){
	    XYSeriesCollection dataset = new XYSeriesCollection();
	    XYSeries series1 = new XYSeries(nameY1);

	    if(X.length != Y1.length)
	    	throw new IllegalArgumentException("createDataSet failed");
	    
		for(int i=0;i<X.length;i++)
			series1.add(X[i], Y1[i]);
		
	    dataset.addSeries(series1);	 
	    return dataset;
	}
	
	private static XYDataset createDataSet(double[] X, double[] Y1, String nameY1, double[] Y2, String nameY2){
	    XYSeriesCollection dataset = new XYSeriesCollection();
	    XYSeries series1 = new XYSeries(nameY1);
	    XYSeries series2 = new XYSeries(nameY2);

	    if(X.length != Y1.length || X.length != Y2.length){
	    	throw new IllegalArgumentException("createDataSet failed");
	    }
	    
		for(int i=0;i<X.length;i++){
			series1.add(X[i], Y1[i]);
			series2.add(X[i], Y2[i]);
		}
	    dataset.addSeries(series1);
	    dataset.addSeries(series2);
	 
	    return dataset;
	}
	
	private void createXYDotsAndLines(XYDataset dataset1, XYDataset dataset2, String domainName, String rangeName,String plotTitle) {
		// Create a single plot containing both the scatter and line
		XYPlot plot = new XYPlot();

		/* SETUP SCATTER */

		// Create the scatter data, renderer, and axis
		XYItemRenderer renderer1 = new XYLineAndShapeRenderer(true, false);   // Lines only
		ValueAxis domain1 = new NumberAxis(domainName);
		ValueAxis range1 = new NumberAxis(rangeName);

		// Set the scatter data, renderer, and axis into plot
		plot.setDataset(0, dataset2);
		plot.setRenderer(0, renderer1);
		plot.setDomainAxis(0, domain1);
		plot.setRangeAxis(0, range1);

		// Map the scatter to the first Domain and first Range
		plot.mapDatasetToDomainAxis(0, 0);
		plot.mapDatasetToRangeAxis(0, 0);

		/* SETUP LINE */

		// Create the line data, renderer, and axis
		XYItemRenderer renderer2 = new XYLineAndShapeRenderer(false, true);   // Shapes only

		// Set the line data, renderer, and axis into plot
		plot.setDataset(1, dataset1);
		plot.setRenderer(1, renderer2);
		//plot.setDomainAxis(1, domain1);
		//plot.setRangeAxis(1, range1);

		// Map the line to the second Domain and second Range
		plot.mapDatasetToDomainAxis(1, 0);
		plot.mapDatasetToRangeAxis(1, 0);
		
		createPlot(plotTitle, plot);
	}	
	
	private void createXYDots(XYDataset dataset1, XYDataset dataset2, String domainName, String rangeName,String plotTitle) {
		// Create a single plot containing both the scatter and line
		XYPlot plot = new XYPlot();

		/* SETUP SCATTER */

		// Create the scatter data, renderer, and axis
		XYItemRenderer renderer1 = new XYLineAndShapeRenderer(false, true);   // Lines only
		ValueAxis domain1 = new NumberAxis(domainName);
		ValueAxis range1 = new NumberAxis(rangeName);

		// Set the scatter data, renderer, and axis into plot
		plot.setDataset(0, dataset2);
		plot.setRenderer(0, renderer1);
		plot.setDomainAxis(0, domain1);
		plot.setRangeAxis(0, range1);

		// Map the scatter to the first Domain and first Range
		plot.mapDatasetToDomainAxis(0, 0);
		plot.mapDatasetToRangeAxis(0, 0);

		/* SETUP LINE */

		// Create the line data, renderer, and axis
		XYItemRenderer renderer2 = new XYLineAndShapeRenderer(false, true);   // Shapes only

		// Set the line data, renderer, and axis into plot
		plot.setDataset(1, dataset1);
		plot.setRenderer(1, renderer2);
		//plot.setDomainAxis(1, domain1);
		//plot.setRangeAxis(1, range1);

		// Map the line to the second Domain and second Range
		plot.mapDatasetToDomainAxis(1, 0);
		plot.mapDatasetToRangeAxis(1, 0);
		
		createPlot(plotTitle, plot);
	}	
	
	private void createPlot(String plotTitle, XYPlot plot){
		// Create the chart with the plot and a legend
		JFreeChart chart = new JFreeChart(plotTitle, JFreeChart.DEFAULT_TITLE_FONT, plot, true);
		
		ChartPanel cp = new ChartPanel(chart);
	    cp.setPreferredSize(new Dimension(750,550));
		
		JPanel jp = new JPanel();
	    jp.add(cp);
		jp.setPreferredSize(new Dimension(800,600));
		
		plotWindow.getContentPane().removeAll();
		plotWindow.setContentPane(jp);
		plotWindow.validate();
		plotWindow.pack();
		plotWindow.setVisible(true);
	}

	void closePlotWindows(){
		if (plotWindow!=null) plotWindow.dispose();
	}
}
