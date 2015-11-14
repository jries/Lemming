package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.Fitter;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizationPrecision3D;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.CentroidFitter;
import org.lemming.plugins.NMSDetector;
import ij.ImagePlus;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;

@SuppressWarnings("rawtypes")
public class ManagerTest2 {

	private Manager pipe;
	private Map<Integer, Store> storeMap;
	private ImagePlus loc_im;
	
	@Before
	public void setUp() throws Exception {
		
        File file = new File(System.getProperty("user.home")+"/ownCloud/storm/TubulinAF647.tif");
        
		if (file.isDirectory()){
        	FolderOpener fo = new FolderOpener();
        	fo.openAsVirtualStack(true);
        	loc_im = fo.openFolder(file.getAbsolutePath());
        }
        
        if (file.isFile()){
        	loc_im = FileInfoVirtualStack.openVirtual(file.getAbsolutePath());
        }
	
	    if (loc_im ==null)
		    throw new Exception("File not found");
		
		ImageLoader tif = new ImageLoader(loc_im);

		NMSDetector peak = new NMSDetector(700,7);
		//Fitter fitter = new QuadraticFitter(10);
		//@SuppressWarnings("unchecked")
		//Fitter fitter = new Fitter(7, Settings.readCSV(System.getProperty("user.home")+"/ownCloud/storm/calTest.csv").get("param"));
		Fitter fitter = new CentroidFitter(10, 700);

		SaveLocalizationPrecision3D saver = new SaveLocalizationPrecision3D(new File(System.getProperty("user.home")+"/ownCloud/storm/fitted2.csv"));
		
		pipe = new Manager();
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(saver);
		
		pipe.linkModules(tif, peak, true);
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,saver);
		storeMap = pipe.getMap();
	}

	@Test
	public void test() {
		pipe.run();
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
	}

}
