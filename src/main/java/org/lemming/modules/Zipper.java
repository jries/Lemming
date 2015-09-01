package org.lemming.modules;

import java.io.File;
import java.io.IOException;
import java.util.zip.ZipFile;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.SingleRunModule;

public class Zipper extends SingleRunModule {
	
	private File file;
	private ZipFile z;

	public Zipper(File file){
		this.file = file;
	}
	
	@Override
	public void beforeRun() {

		try {
			z = new ZipFile(file);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public Element process(Element data) {
		return null;
	}
	
	@Override
	public void afterRun() {
		try {
			z.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public boolean check() {
		return true;
	}
	
}

