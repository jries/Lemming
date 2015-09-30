package org.lemming.tests;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.DataTable;
import org.lemming.modules.ImageLoader;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.PeakFinder;
import org.lemming.plugins.QuadraticFitter;

import ij.ImagePlus;

public class ExtendableTableTest2 {
	
	private Manager pipe;
	private DataTable dt;

	@SuppressWarnings("rawtypes")
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		ImageLoader tif = new ImageLoader(new ImagePlus("/home/ronny/ownCloud/storm/p500ast.tif"));
		PeakFinder peak = new PeakFinder(700,4);
		QuadraticFitter fitter = new QuadraticFitter(100,10);
		dt = new DataTable(); 
		
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(dt);
		
		pipe.linkModules(tif, peak, true);
		pipe.linkModules(tif,fitter); // first images
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,dt);
		
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println(dt.getTable().columnNames().toString());
		System.out.println(dt.getTable().getNumberOfRows());
	}

}
