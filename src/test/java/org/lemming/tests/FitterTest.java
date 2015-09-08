package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveFittedLocalizations;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Settings;
import org.lemming.plugins.AstigFitter;
import org.lemming.plugins.PeakFinder;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class FitterTest {

	private Pipeline pipe;
	private Map<Integer,Store> storeMap = new HashMap<>();
	private int hash1;
	private int hash2;


	
	@SuppressWarnings("unchecked")
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");
		FastStore frames1 = new FastStore();
		FastStore frames2 = new FastStore();
		hash1=frames1.hashCode();
		hash2=frames2.hashCode();
		
		ImageLoader tif = new ImageLoader(new ImagePlus("/home/ronny/ownCloud/storm/p500ast.tif"));
		tif.setOutput(frames1);
		tif.setOutput(frames2);
		storeMap.put(hash1, frames1);
		storeMap.put(hash2, frames2);
		pipe.add(tif);
		
		FastStore locs1 = new FastStore();
		FastStore locs2 = new FastStore();
		PeakFinder peak = new PeakFinder(700,4);
		peak.setInput(frames2);
		peak.setOutput(locs1);
		peak.setOutput(locs2);
		pipe.add(peak);
		
		FastStore fitlocs = new FastStore();
		AstigFitter fitter = new AstigFitter(60,10, Settings.readProps("/home/ronny/ownCloud/storm/Settings.properties"));
		fitter.setInput(frames1);
		fitter.setInput(locs1);
		fitter.setOutput( fitlocs);
		pipe.add(fitter);
		
		FastStore unpackedlocs = new FastStore();
		UnpackElements unpacker = new UnpackElements();
		unpacker.setInput(locs2);
		unpacker.setOutput(unpackedlocs);
		pipe.add(unpacker);
		
		SaveFittedLocalizations saver = new SaveFittedLocalizations(new File("/home/ronny/Bilder/fitted.csv"));
		//saver = new SaveFittedLocalizations(new File("/Users/ronny/Documents/fitted.csv"));
		saver.setInput(fitlocs);
		pipe.add(saver);
		
		SaveLocalizations saver2 = new SaveLocalizations(new File("/home/ronny/Bilder/outOrig.csv"));
		//saver2 = new SaveLocalizations(new File("/Users/ronny/Documents/outOrig.csv"));
		saver2.setInput(unpackedlocs);
		pipe.add(saver2);
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,storeMap.get(hash1).isEmpty());
		assertEquals(true,storeMap.get(hash2).isEmpty());
	}

}
