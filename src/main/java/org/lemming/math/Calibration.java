package org.lemming.math;

import java.awt.Dimension;
import java.awt.Toolkit;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import javax.swing.JFrame;
import javax.swing.JPanel;

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

/**
 *  a class for handling the calibration process
 * 
 * @author Ronny Sczech
 *
 */
class Calibration {

	private static final int INDEX_WX = 0;
	private static final int INDEX_WY = 1;
	private static final int INDEX_AX = 2;
	private static final int INDEX_AY = 3;
	private static final int INDEX_BX = 4;
	private static final int INDEX_BY = 5;
	private static final int INDEX_C = 6;
	private static final int INDEX_D = 7;
	private static final int INDEX_Mp = 8;
	private static final int PARAM_LENGTH = 9;
	private JFrame plotWindow;

	private double[] zgrid;										// z positions of the slices in the stack
	private double[] Wx;
	private double[] Wy;
	private double[] Calibcurve;						// 1D and 2D fit results
	private double[] curveWx;
	private double[] curveWy;							// quadratically fitted curves
	private int nSlice=1;

	private double[] param;
	
	public Calibration(){
		initialize();	
	}
	
	public Calibration(double[] zgrid, double[] Wx, double[] Wy, double[] curveWx, double[] curveWy, double[] Calibcurve, double[] param){
		nSlice = zgrid.length;
		this.zgrid = zgrid;						// z position of the frames
		this.Wx = Wx;							// width in x of the PSF
		this.Wy = Wy;							// width in y of the PSF
		this.Calibcurve = Calibcurve;
		this.curveWx = curveWx;					// value of the calibration on X
		this.curveWy = curveWy;					// value of the calibration on Y
		this.param = param;						// parameters of the calibration on X and Y
		plotWindow = new JFrame();
		plotWindow.setPreferredSize(new Dimension(800,600));
		Dimension dim = Toolkit.getDefaultToolkit().getScreenSize();
		plotWindow.setLocation(dim.width/2-400, dim.height/2-300);
		plotWindow.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	}
	
	private void initialize(){
		zgrid = new double[nSlice];						// z position of the frames
		Wx = new double[nSlice];						// width in x of the PSF
		Wy = new double[nSlice];						// width in y of the PSF
		Calibcurve = new double[nSlice];
		curveWx = new double[nSlice];					// value of the calibration on X
		curveWy = new double[nSlice];					// value of the calibration on Y
		param = new double[PARAM_LENGTH];				// parameters of the calibration on X and Y
	}
	
	///////////////////////////////////////// Setters and getters
	public void setZgrid(double[] zgrid){
		this.zgrid = zgrid;
	}
	public double[] getZgrid(){
		return zgrid;
	}
	
	public void setWx(double[] Wx){
		this.Wx = Wx;
	}
	public double[] getWx(){
		return Wx;
	}

	public void setWy(double[] Wy){
		this.Wy = Wy;
	}
	public double[] getWy(){
		return Wy;
	}
	
	public void setcurveWx(double[] curveWx){
		this.curveWx = curveWx;
	}
	public double[] getcurveWx(){
		return curveWx;
	}

	public void setcurveWy(double[] curveWy){
		this.curveWy = curveWy;
	}
	public double[] getcurveWy(){
		return curveWy;
	}

	public void setCalibcurve(double[] Calibcurve){
		this.Calibcurve = Calibcurve;
	}
	public double[] getCalibcurve(){
		return Calibcurve;
	}
	
	///////////////////////////////////////// Plot
	public void plot(double[] W1, double[] W2, String title){
		if(W1.length > 0 && W2.length>0){
        	createXYDots(createDataSet(zgrid, W1, "Width in x", W2, "Width in y"), "Z (nm)", "Width", title);
		}
	}	

	public void plotWxWyFitCurves(){
		if(Wx.length > 0 && Wy.length>0 && curveWy.length>0 && curveWy.length>0){
			createXYDotsAndLines(createDataSet(zgrid, Wx, "Width in X", Wy, "Width in Y"),
					createDataSet(zgrid, curveWx, "Fitted width in X", curveWy, "Fitted width in Y"), "Z (nm)", "Width", "Width of Elliptical Gaussian");
		}
	}


	///////////////////////////////////////// Save
	public void saveAsCSV(String path){
		csvWriter w = new csvWriter(new File(path));
		w.process("zGrid, Wx, Wy, Curve, curveWx, CurveWy \n");
 	    for (int i=0; i< zgrid.length;i++){
 	    	String s = "" + zgrid[i] + ", " + Wx[i] + ", " + Wy[i] + ", " + Calibcurve[i] + ", " +  curveWx[i] + ", " + curveWy[i] + "\n";
 	    	w.process(s);
 	    }
 	   w.process("--\n");
 	   String ps = "";
 	   for (int j=0;j<PARAM_LENGTH;j++)
 		   ps += param[j] + ", ";
 	   ps = ps.substring(0,ps.length()-3);
 	   ps += "\n";
 	   w.process(ps);	   
 	   w.close();
	}
	
	public void readCSV(String path){
		final Locale curLocale = Locale.getDefault();
		final Locale usLocale = new Locale("en", "US"); // setting us locale
		Locale.setDefault(usLocale);
		
		List<String> list = new ArrayList<>();
		
		try {
			BufferedReader br = new BufferedReader(new FileReader(path));

			String line;
			String[] s;
			br.readLine();
			while ((line=br.readLine())!=null){
				if (line.contains("--")) break;
				list.add(line);
			}
			
			nSlice = list.size();
			initialize();
			
			if ((line=br.readLine())!=null){
				s = line.split(",");
				for (int i = 0; i < s.length; i++)
					s[i] = s[i].trim();
				param = new double[]{Double.parseDouble(s[0]),
						Double.parseDouble(s[1]),
						Double.parseDouble(s[2]),
						Double.parseDouble(s[3]),
						Double.parseDouble(s[4]),
						Double.parseDouble(s[5]),
						Double.parseDouble(s[6]),
						Double.parseDouble(s[7]),
						Double.parseDouble(s[8])};
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		for (int i=0; i<nSlice;i++){
			String[] s = list.get(i).split(",");
			for (int j = 0; j < s.length; j++)
				s[j] = s[j].trim();
			zgrid[i] = Double.parseDouble(s[0]);
			Wx[i] = Double.parseDouble(s[1]);
			Wy[i] = Double.parseDouble(s[2]);
			Calibcurve[i] = Double.parseDouble(s[3]);
			curveWx[i] = Double.parseDouble(s[4]);
			curveWy[i] = Double.parseDouble(s[5]);
		}
		
		Locale.setDefault(curLocale);
	}

	public double getValueWx(double z) {
		double b = (z-param[INDEX_C]-param[INDEX_Mp])/param[INDEX_D];
		return param[INDEX_WX]*Math.sqrt(1+b*b+param[INDEX_AX]*b*b*b+param[INDEX_BX]*b*b*b*b);
	}

	public double getValueWy(double z) {
		double b = (z+param[INDEX_C]-param[INDEX_Mp])/param[INDEX_D];
		return param[INDEX_WY]*Math.sqrt(1+b*b+param[INDEX_AY]*b*b*b+param[INDEX_BY]*b*b*b*b);
	}

	public double[] getparam() {
		return param;
	}
	
	public void setparam(double[] param){
		this.param = param;
	}
	
	public void closePlotWindows(){
		if (plotWindow!=null) plotWindow.dispose();
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
	
	private void createXYDots(XYDataset xy, String domainName, String rangeName, String plotTitle){															////////////////////////////// change to be less redundant with previous function
		// Create a single plot
		XYPlot plot = new XYPlot();

		/* SETUP SCATTER */

		// Create the scatter data, renderer, and axis
		XYItemRenderer renderer = new XYLineAndShapeRenderer(false, true);   // Shapes only
		ValueAxis domain = new NumberAxis(domainName);
		ValueAxis range = new NumberAxis(rangeName);

		// Set the scatter data, renderer, and axis into plot
		plot.setDataset(0, xy);
		plot.setRenderer(0, renderer);
		plot.setDomainAxis(0, domain);
		plot.setRangeAxis(0, range);

		// Map the scatter to the first Domain and first Range
		plot.mapDatasetToDomainAxis(0, 0);
		plot.mapDatasetToRangeAxis(0, 0);
		createPlot(plotTitle, plot);
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
		plotWindow.revalidate();
		plotWindow.repaint();
		plotWindow.pack();
		plotWindow.setVisible(true);
	}
	
	private class csvWriter {

		private final Locale curLocale;
		private FileWriter w;

		csvWriter(File file) {
			this.curLocale = Locale.getDefault();
			final Locale usLocale = new Locale("en", "US"); // setting us locale
			Locale.setDefault(usLocale);
			try {
				w = new FileWriter(file);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public void process(String p){
			try {
				w.write(p);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		public void close(){
			try {
				w.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			Locale.setDefault(curLocale);
		}
	}
	
}
