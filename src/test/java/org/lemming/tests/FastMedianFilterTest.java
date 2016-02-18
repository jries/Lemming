package org.lemming.tests;

import static org.junit.Assert.*;

import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveImages;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.FastMedianFilter;
import org.lemming.tools.LemmingUtils;

import ij.ImagePlus;

@SuppressWarnings("rawtypes")
public class FastMedianFilterTest {

	private Manager pipe;
	private ImageLoader tif;
	private FastMedianFilter fmf;
	private SaveImages saver;
	private Map<Integer, Store> map;
	
	@Before
	public void setUp() throws Exception {
		pipe = new Manager();
		
		tif = new ImageLoader<>(new ImagePlus("/Users/ronny/Documents/TubulinAF647.tif"), LemmingUtils.readCameraSettings("camera.props"));
		pipe.add(tif);
		
		fmf = new FastMedianFilter(50,true);

		pipe.add(fmf);
		
		saver = new SaveImages("/home/ronny/Bilder/out.tif");
		pipe.add(saver);
		
		pipe.linkModules(tif, fmf);
		pipe.linkModules(fmf, saver);
		
		map = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true, map.values().iterator().next().isEmpty());
	}

}
