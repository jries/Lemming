package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.AstigFitter;
import org.lemming.modules.Fitter;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.PeakFinder;
import org.lemming.modules.SaveFittedLocalizations;
import org.lemming.modules.SaveLocalizations;
import org.lemming.modules.StoreSplitter;
import org.lemming.modules.UnpackElements;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Settings;

@SuppressWarnings("rawtypes")
public class FitterTest {

	private Pipeline pipe;
	private FastStore fitlocs;
	private FastStore frames;
	private AbstractModule tif;
	private FastStore localizations;
	private PeakFinder peak;
	private Fitter fitter;
	private SaveFittedLocalizations saver;
	private StoreSplitter splitter2;
	private FastStore locs1;
	private FastStore locs2;
	private SaveLocalizations saver2;
	private FastStore frames1;
	private Settings settings;
	private StoreSplitter splitter1;
	private FastStore frames2;
	private UnpackElements unpacker;
	private FastStore unpackedlocs;

	
	@Before
	public void setUp() throws Exception {
		settings = new Settings();//  global settings
		pipe = new Pipeline("test");
		frames = new FastStore();
		tif = new IJTiffLoader("/home/ronny/ownCloud/storm/p500ast.tif");
		//tif = new IJTiffLoader("/Users/ronny/Documents/p500ast.tif");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		splitter1 = new StoreSplitter();
		splitter1.setInput("frames",frames);
		frames1= new FastStore();
		frames2= new FastStore();
		splitter1.setOutput("frames1",frames1);
		splitter1.setOutput("frames2",frames2);
		pipe.add(splitter1);
		
		localizations = new FastStore();
		peak = new PeakFinder(settings, 700,4);
		peak.setInput("frames", frames2);
		peak.setOutput("locs", localizations);
		pipe.add(peak);
		
		splitter2 = new StoreSplitter();
		splitter2.setInput("locs",localizations);
		locs1= new FastStore();
		locs2= new FastStore();
		splitter2.setOutput("locs1",locs1);
		splitter2.setOutput("locs2",locs2);
		pipe.add(splitter2);	
		
		fitlocs = new FastStore();
		fitter = new AstigFitter(60,10, Settings.readCSV("/home/ronny/ownCloud/storm/calibNewer.csv"));
		fitter.setInput("frames1", frames1);
		fitter.setInput("locs1", locs1);
		fitter.setIterator("frames1");
		fitter.setOutput("fitLocs", fitlocs);
		pipe.add(fitter);
		
		unpackedlocs = new FastStore();
		unpacker = new UnpackElements();
		unpacker.setInput("locs2", locs2);
		unpacker.setOutput("unpacked", unpackedlocs);
		pipe.add(unpacker);
		
		saver = new SaveFittedLocalizations(new File("/home/ronny/Bilder/fitted.csv"));
		//saver = new SaveFittedLocalizations(new File("/Users/ronny/Documents/fitted.csv"));
		
		saver.setInput("fitlocs", fitlocs);
		pipe.add(saver);
		
		saver2 = new SaveLocalizations(new File("/home/ronny/Bilder/outOrig.csv"));
		//saver2 = new SaveLocalizations(new File("/Users/ronny/Documents/outOrig.csv"));
		saver2.setInput("fitlocs", unpackedlocs);
		pipe.add(saver2);
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,locs1.isEmpty());
		assertEquals(true,frames.isEmpty());
	}

}
