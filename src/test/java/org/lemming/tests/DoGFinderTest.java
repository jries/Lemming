package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.DoGFinder;
import org.lemming.modules.IJTiffLoader;
import org.lemming.modules.SaveLocalizations;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Pipeline;

@SuppressWarnings("rawtypes")
public class DoGFinderTest {

	private Pipeline pipe;
	private FastStore frames;
	private IJTiffLoader tif;
	private FastStore localizations;
	private DoGFinder peak;
	private SaveLocalizations saver;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Pipeline();	
		
		frames = new FastStore();
		//tif = new IJTiffLoader("/home/ronny/Bilder/TubulinAF647.tif");
		tif = new IJTiffLoader("/Users/ronny/Documents/storm/sequence.tif");
		tif.setOutput("frames",frames);
		pipe.add(tif);
		
		localizations = new FastStore();
		peak = new DoGFinder(6,6);
		peak.setInput("frames", frames);
		peak.setOutput("locs", localizations);
		pipe.add(peak);
		
		//saver = new SaveLocalizations(new File("/home/ronny/Bilder/out.csv"));
		saver = new SaveLocalizations(new File("/Users/ronny/Documents/storm/dogfinder.csv"));
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
