package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.PeakFinder;
import org.lemming.modules.SaveLocalizations;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.pipeline.Settings;

@SuppressWarnings("rawtypes")
public class PeakFinderTest {

	private Pipeline pipe;
	private FastStore frames;
	private IJTiffLoader tif;
	private FastStore localizations;
	private PeakFinder peak;
	private SaveLocalizations saver;
	private Settings settings;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");	
		settings = new Settings();
		frames = new FastStore();
		//tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		tif = new IJTiffLoader("/Users/ronny/Documents/storm/sequence.tif");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		localizations = new FastStore();
		peak = new PeakFinder(settings,400,4);
		peak.setInput("frames", frames);
		peak.setOutput("locs", localizations);
		pipe.add(peak);
		
		//saver = new SaveLocalizations(new File("/home/ronny/Bilder/out.csv"));
		saver = new SaveLocalizations(new File("/Users/ronny/Documents/storm/peakfinder.csv"));
		saver.setInput("locs", localizations);
		pipe.add(saver);
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println("");		
		assertEquals(true,frames.isEmpty());
	}

}
