package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveFittedLocalizations;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.Manager;
import org.lemming.pipeline.Settings;
import org.lemming.plugins.AstigFitter;
import org.lemming.plugins.PeakFinder;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class ManagerTest {

	private Manager pipe;
	private Map<Integer, Store> storeMap;
	
	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		ImageLoader tif = new ImageLoader(new ImagePlus("/home/ronny/ownCloud/storm/p500ast.tif"));
		PeakFinder peak = new PeakFinder(700,4);
		AstigFitter fitter = new AstigFitter(60,10, Settings.readProps("/home/ronny/ownCloud/storm/Settings.properties"));
		UnpackElements unpacker = new UnpackElements();
		SaveFittedLocalizations saver = new SaveFittedLocalizations(new File("/home/ronny/Bilder/fitted.csv"));
		//saver = new SaveFittedLocalizations(new File("/Users/ronny/Documents/fitted.csv"));
		SaveLocalizations saver2 = new SaveLocalizations(new File("/home/ronny/Bilder/outOrig.csv"));
		//saver2 = new SaveLocalizations(new File("/Users/ronny/Documents/outOrig.csv"));
		
		
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(unpacker);
		pipe.add(saver);
		pipe.add(saver2);
		
		pipe.linkModules(tif, peak, true);
		pipe.linkModules(tif,fitter); // first images
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,saver);
		pipe.linkModules(peak,unpacker);
		pipe.linkModules(unpacker,saver2);
		storeMap = pipe.get();
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
	}

}
