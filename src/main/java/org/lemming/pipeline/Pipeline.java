package org.lemming.pipeline;

import java.io.File;
import java.io.IOException;
import java.util.LinkedList;
import java.util.zip.ZipFile;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Store;

public class Pipeline implements Runnable {

	private ThreadGroup group;
	private LinkedList<Thread> pipe;
	public Store zipStore = new FastStore();
	/**
	 * 
	 */
	public Pipeline(String projectName){
		this.group = new ThreadGroup("Pipe");
		this.pipe = new LinkedList<>();
		/*ZipModule zip = new ZipModule(new File(IJ.getDirectory("home")+projectName));
		zip.setInput("zip", zipStore);
		pipe.addLast(new Thread(group, zip, "Zipper"));*/
	}
	
	/**
	 * @param module - Runnable to add
	 */
	public void add(AbstractModule module){
		pipe.addLast(new Thread(group, module, module.getClass().getSimpleName()));
	}
	
	/**
	 * @param module - Module to run as sequential
	 */
	public void addSequential(AbstractModule module){
		if (!pipe.isEmpty()) run();
		module.run();
	
		while(module.isRunning()){ //wait until store is filled up completely
			pause(10);
		}
		pipe.clear();
	}

	@Override
	public void run() {
		
		for(Thread starter:pipe)
			starter.start();
		
		for(Thread joiner:pipe){
			try {
				joiner.join();
			} catch (InterruptedException e) {
				System.err.println(e.getMessage());
			}
		}
		group.interrupt();
	}
	
	private static void pause(long ms){
		try {
			Thread.sleep(ms);
		} catch (InterruptedException e) {
			System.err.println(e.getMessage());
		}
	}
	
	private class ZipModule extends SingleRunModule {
		
		private File file;
		private ZipFile z;

		public ZipModule(File file){
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
			// TODO Auto-generated method stub
			return false;
		}
		
	}

}
