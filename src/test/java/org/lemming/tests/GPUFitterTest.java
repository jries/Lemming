package org.lemming.tests;

import static org.junit.Assert.*;

import java.io.File;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.lemming.interfaces.Store;
import org.lemming.modules.ImageLoader;
import org.lemming.modules.SaveLocalizationPrecision3D;
import org.lemming.pipeline.AbstractModule;
import org.lemming.pipeline.Manager;
import org.lemming.plugins.MLE_Fitter;
import org.lemming.plugins.NMSDetector;
import ij.ImagePlus;
import ij.plugin.FileInfoVirtualStack;
import ij.plugin.FolderOpener;

@SuppressWarnings("rawtypes")
public class GPUFitterTest {

	private Manager pipe;
	private Map<Integer, Store> storeMap;
	private ImagePlus loc_im;
	
	@Before
	public void setUp() throws Exception {
		
        //File file = new File(System.getProperty("user.home")+"/Documents/storm/experiment3D.tif");
        File file = new File("D:/Images/DRG_KO_5_1.tif");
        
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
		
		AbstractModule tif = new ImageLoader(loc_im);
		AbstractModule peak = new NMSDetector(5000,4);
		//AbstractModule peak = new NMSDetector(2000,5); DRG_WT_MT_A647_1.tif
		AbstractModule fitter = new MLE_Fitter<>(5);
		AbstractModule saver = new SaveLocalizationPrecision3D(new File("D:/Images/DRG_KO_5_1.csv"));
		
		pipe = new Manager();
		pipe.add(tif);
		pipe.add(peak);
		pipe.add(fitter);
		pipe.add(saver);
		
		pipe.linkModules(tif, peak, true, loc_im.getStackSize());
		pipe.linkModules(peak,fitter);
		pipe.linkModules(fitter,saver,false, 256);
		storeMap = pipe.getMap();
	}

	@Test
	public void test() {
		long start = System.currentTimeMillis();
		pipe.run();
		long end = System.currentTimeMillis();
		System.out.println("Overall: " + (end-start) +"ms");
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
		assertEquals(true,storeMap.values().iterator().next().isEmpty());
	}

}
