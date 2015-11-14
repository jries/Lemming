package org.lemming.tests;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveImages;
import org.lemming.pipeline.FastStore;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.FastMedianFilter;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class FastMedianFilterTest {

	private Manager pipe;
	private FastStore frames;
	private ImageLoader tif;
	private FastMedianFilter fmf;
	private SaveImages saver;
	private FastStore filtered;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		frames = new FastStore();
		tif = new ImageLoader(new ImagePlus("/Users/ronny/Documents/TubulinAF647.tif"));
		tif.setOutput(frames);
		pipe.add(tif);
		
		filtered = new FastStore();
		fmf = new FastMedianFilter(50,true);
		fmf.setInput(frames);
		fmf.setOutput(filtered);
		pipe.add(fmf);
		
		saver = new SaveImages("/home/ronny/Bilder/out.tif");
		saver.setInput(filtered);
		pipe.add(saver);
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,frames.isEmpty());
	}

}
