package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.DoGFinder;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizations;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class DoGFinderTest {

	private Pipeline pipe;
	private FastStore frames;
	private ImageLoader tif;
	private FastStore localizations;
	private DoGFinder peak;
	private SaveLocalizations saver;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline("test");
		
		frames = new FastStore();
		//tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		tif = new ImageLoader(new ImagePlus("/Users/ronny/Documents/TubulinAF647.tif"));
		tif.setOutput(frames);
		pipe.add(tif);
		
		localizations = new FastStore();
		peak = new DoGFinder(6,6);
		peak.setInput( frames);
		peak.setOutput(localizations);
		pipe.add(peak);
		
		//saver = new SaveLocalizations(new File("/home/ronny/Bilder/out.csv"));
		saver = new SaveLocalizations(new File("/Users/ronny/Documents/storm/dogfinder.csv"));
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
