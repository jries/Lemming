package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.math.FitterType;
import org.lemming.modules.Fitter;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.PeakFinder;
import org.lemming.modules.SaveFittedLocalizations;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.StoreSplitter;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Store;

@SuppressWarnings("rawtypes")
public class FitterTest {

	private Pipeline pipe;
	private FastStore fitlocs;
	private FastStore frames;
	private AbstractModule tif;
	private FastStore localizations;
	private PeakFinder peak;
	private Fitter fitter;
	private StoreSplitter splitter;
	private FastStore frames1;
	private FastStore frames2;
	private SaveFittedLocalizations saver;
	private StoreSplitter splitter2;
	private FastStore locs1;
	private FastStore locs2;
	private SaveLocalizations saver2;

	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();
		frames = new FastStore();
		tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		//tif = new IJTiffLoader("/Users/ronny/Documents/TubulinAF647.tif");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		frames1 = new FastStore();
		frames2 = new FastStore();
		splitter = new StoreSplitter();
		Map<String,Store> storeMap = new HashMap<>();
		splitter.setInput("frames", frames);
		storeMap.put("frames1", frames1);
		storeMap.put("frames2", frames2);
		splitter.setOutputs(storeMap);
		pipe.add(splitter);	
		
		localizations = new FastStore();
		peak = new PeakFinder(700,4);
		peak.setInput("frames", frames1);
		peak.setOutput("locs", localizations);
		pipe.addSequential(peak);
		
		splitter2 = new StoreSplitter();
		splitter2.setInput("locs",localizations);
		locs1= new FastStore();
		locs2= new FastStore();
		splitter2.setOutput("locs1",locs1);
		splitter2.setOutput("locs2",locs2);
		pipe.add(splitter2);	
		
		fitlocs = new FastStore();
		fitter = new Fitter(FitterType.QUADRATIC,5);
		fitter.setInput("frames", frames2);
		fitter.setInput("locs", locs1);
		fitter.setIterator("frames");
		fitter.setOutput("fitLocs", fitlocs);
		pipe.add(fitter);
		
		saver = new SaveFittedLocalizations(new File("/home/ronny/Bilder/fitted.csv"));
		saver.setInput("fitlocs", fitlocs);
		pipe.add(saver);
		
		saver2 = new SaveLocalizations(new File("/home/ronny/Bilder/out.csv"));
		saver2.setInput("fitlocs", locs2);
		pipe.add(saver2);
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,frames2.isEmpty());
		assertEquals(true,frames1.isEmpty());
	}

}
