package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizations;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;
import org.lemming.plugins.PeakFinder;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class PeakFinderTest {

	private Pipeline pipe;
	private FastStore frames;
	private ImageLoader tif;
	private FastStore localizations;
	private PeakFinder peak;
	private SaveLocalizations saver;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");	
		frames = new FastStore();
		//tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		tif = new ImageLoader(new ImagePlus("/home/ronny/ownCloud/storm/p500ast.tif"));
		tif.setOutput(frames);
		pipe.add(tif);
		
		localizations = new FastStore();
		peak = new PeakFinder(400,4);
		peak.setInput(frames);
		peak.setOutput(localizations);
		pipe.add(peak);
		
		//saver = new SaveLocalizations(new File("/home/ronny/Bilder/out.csv"));
		saver = new SaveLocalizations(new File("/Users/ronny/Documents/storm/peakfinder.csv"));
		saver.setInput(localizations);
		pipe.add(saver);
	}

	@Test
	public void test() {
		pipe.run();
		System.out.println("");		
		assertEquals(true,frames.isEmpty());
	}

}
